(ns flare.neanderthal-ops
  (:use [uncomplicate.neanderthal core native])
  (:require [flare.graph :as graph]
            [flare.core :as flare]
            [flare.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.real :as real]
            [flare.cache-pool :as cache-pool]
            [uncomplicate.neanderthal.vect-math :as vect-math]
            [flare.computation-graph :as cg])
  (:import [flare.node Node]
           [org.apache.commons.math3.util FastMath]
           [org.apache.commons.math3.analysis.function Sigmoid]
           [java.util LinkedList]
           [java.util.concurrent.atomic AtomicLong]
           [clojure.lang IFn$DD IFn$ODD IFn$DDD IFn$ODDD]))

(defn -mk-matrix 
  ([rows cols] (dge rows cols {:layout :row}))
  ([rows cols data] (dge rows cols (flatten data) {:layout :row})))

(defn ^:private valid-shape? [shape]
  (<= (count shape) 2))

(defn ^:private ensure-valid-shape?! [shape]
  (when-not (valid-shape? shape)
    (throw (ex-info "Must use a vector (1d) or matrix (2d) shape"
                    {:shape shape}))))

(defn -zeros [shape]
  (case (count shape)
    1 (dv (first shape))
    2 (-mk-matrix (first shape) (second shape))
    (throw (ex-info "Unallowed shape for neanderthal"
                    {:shape (vec shape)}))))

(defrecord SumTensorOp []
  cg/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [^Node n input-nodes] (ensure-valid-shape?! (.shape n))))
  (forward-node-pass! [this node]
    (let [^Node node node
          output (.value node)
          inputs (mapv #(.value ^Node %) (.children node))]
      (copy! (first inputs) output)
      (doseq [input (rest inputs)]
        ;; output += input
        (np/axpy! input output)))
    node)
  (backward-node-pass! [this node]
    ;; x = x1 + ... + xn
    ;; dxi/dt = dx/dt * 1
    ;; so just add output grad to inputs
    (let [^Node node node
          df_dt (.grad node)
          input-grads (map #(.grad ^Node %) (.children node))]
      (doseq [input-grad input-grads :when input-grad]
        ;; input-grad += df_dt
        (np/axpy! df_dt input-grad)))
    node))

(defrecord MultTensorOp []
  cg/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [^Node n input-nodes]
      (ensure-valid-shape?! (.shape n)))
    (when-not (= 2 (count input-nodes))
      (throw (ex-info "Must have two arguments to MultTensorOp"))))
  (forward-node-pass! [this node]
    (let [^Node node node
          out (.value node)
          _ (when-not out (throw (ex-info "Missing .value" {:node node})))
          [a b] (map #(.value ^Node %) (.children node))]
      ;; mm!/mv! will add, so need to clear first
      (if (vctr? b)
        ;; out = a b
        (mv! 1.0 a b 0.0 out)
        (mm! 1.0 a b 0.0 out)))
    node)
  (backward-node-pass! [this node]
    ;; Z[m,n] = X[m,k] Y[k,n]
    ;; dX/dt[m,k] = dZ/dt[m,n] Y^T [n, k]
    (let [^Node node node
          dZ (.grad node)
          [^Node nX ^Node nY] (.children node)
          X (.value nX)
          Y (.value nY)
          dX (.grad nX)
          dY (.grad nY)]
      ;; update dX
      (when dX
        (if (matrix? Y)
          ;; dX += dZ Y'
          (np/mm! 1.0 dZ (trans Y) dX)
          (np/rk! dZ Y dX)))
      ;; update dY
      (when dY
        ;; dY += X' dZ
        (if (matrix? Y)
          (np/mm! 1.0 (trans X) dZ dY)
          (np/mv! 1.0 (trans X) dZ dY))))
    node)
  compute/BatchTensorOp
  (batch-signature [this node]
    ;; only batch when right-hand side is a (row) vector
    ;; and the left-hand-side is the same value
    ;; so we can do a Matrix * Matrix operation
    ;; instead of a Matrix * Vector
    (let [[a b] (:children node)
          [nrows ncols] (:shape b)]
      ;; b is either vector or (row) vector
      (when (or (nil? ncols) (= ncols 1))
        [::mult (:ref-name a) (:shape b)])))
  (batch-forward-node-pass! [this sig nodes]
    (let [[_ m-ref v-shape] sig
          M (-> nodes first :children first)
          num-rows (first v-shape)
          num-cols (count nodes)
          cached (:batched-cache (first nodes))
          new-input (if cached
                      (first cached)
                      (-mk-matrix num-rows num-cols))]
      (when-not (= m-ref (:ref-name M))
        (throw (ex-info "Bad M ref" {:m-ref m-ref :M M})))
      ;; copy columns to new matrix
      (loop [j 0 nodes nodes]
        (when-let [node (first nodes)]
          (let [b (-> node :children second)]
            (transfer! (:value b)
                       (submatrix new-input 0 j num-rows 1))
            (recur (inc j) (next nodes)))))
      ;; perfrom mm, copy columns to outputs
      (let [result (if cached (second cached)
                       (-mk-matrix (-> M :shape first) (count nodes)))]
        (scal! 0.0 result)
        (mm! 1.0 (:value M) new-input 1.0 result)
        (loop [nodes nodes cols (cols result)]
          (when-let [n (first nodes)]
            (transfer! (first cols) (:value n))
            (recur (next nodes) (next cols))))
        (if cached
          nodes
          (concat
           [(assoc (first nodes) :batched-cache [new-input result])]
           (rest nodes)))))))

(defrecord SumElemsTensorOp []
  cg/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [^Node n input-nodes]
      (ensure-valid-shape?! (.shape n))))
  (forward-node-pass! [this node]
    (let [input (-> node :children first :value)
          s (double (sum input))]
      (if (vctr? input)
        (real/entry! (:value node) 0 s)
        (real/entry! (:value node) 0 0 s)))
    node)
  (backward-node-pass! [this node]
    (let [in-grad (-> node :grad first double)
          grad (-> node :children first :grad)]
      (alter! grad (fn ^double [^double x] (+ x in-grad))))))


(defrecord SqueezeTensorOp []
  cg/TensorOp
  (ensure-valid?! [this input-nodes]
    (when-not (= 2 (count (.shape ^Node (first input-nodes))))
      (throw (ex-info "Need matrix shape"
                      {:shape (.shape ^Node (first input-nodes))}))))
  (forward-node-pass! [this node]
    (let [^Node node node
          output (.value node)
          ^Node input (-> node .children first)]
      (copy! (view-vctr (.value input)) output))
    node)
  (backward-node-pass! [this node]
    (let [out-grad (:grad node)
          in-grad (-> node :children first :grad)]
      (when in-grad
        (copy! out-grad (view-vctr in-grad))))
    node))

(defrecord StrechTensorOp []
  cg/TensorOp
  (ensure-valid?! [this input-nodes]
    (let [shape (:shape (first input-nodes))]
      (when-not (flare/vector-shape? shape)
        (throw (ex-info "Need vector shape" {:shape shape})))))
  (forward-node-pass! [this node]
    (let [^Node node node
          output (.value node)
          ^Node input (-> node .children first)]
      (copy! (.value input) (view-vctr output)))
    node)
  (backward-node-pass! [this node]
    (let [^Node node node
          out-grad (.grad node)
          ^Node in (-> node .children first)]
      (when-let [in-grad (.grad in)]
        (axpy! out-grad (view-ge in-grad))))
    node))

(defn soft-max! [scores probs!]
  ;; copy scores + exp in place
  (alter! probs!
          (fn ^double [^long idx ^double x] 
            (FastMath/exp (real/entry scores idx))))
  ;; normalize
  (let [Z (double (asum probs!))]
    (scal! (/ 1.0 Z) probs!)
    probs!))

(defrecord CrossEntropyLossTensorOp []
  cg/TensorOp
  (ensure-valid?! [this [activations label]]
    true)
  (forward-node-pass! [this node]
    (let [^Node node node
          [^Node activations-node ^Node label-node] (.children node)
          activations (.value activations-node)
          len (-> activations-node .shape first)
          probs (soft-max! activations (dv len))
          label (-> label-node .value (entry 0) long)
          correct-prob (real/entry probs label)]
      (when (>= label (dim probs))
        (throw (ex-info "Label index out-of-bounds"
                        {:label label
                         :dim (dim activations)})))
      (real/entry! (.value node) 0 (- (Math/log correct-prob)))
      (assoc node ::probs probs)))
  (backward-node-pass! [this node]
    (let [^Node node node
          [^Node activations-node ^Node label-node] (.children node)
          probs (::probs node)
          loss-grad-val (-> node .grad (real/entry 0))]
      ;; l = (i == label) log(pi)
      (when-let [g (.grad label-node)]
        (throw (ex-info "Don't support label differentiation")))
      ;; d pi / dt = dl/dt (1.0/pi)
      (let [gold-idx (long (first (.value label-node)))
            activations (.value activations-node)]
        (when-let [act-grad (.grad activations-node)]
          (alter! act-grad
           (fn ^double [^long idx ^double cur]
             (+ cur
                (* loss-grad-val
                   (- (real/entry probs idx)
                      (if (= idx gold-idx) 1.0 0.0)))))))))
    node))

#_(defn ^:static hadamard [out! x y reset?]
  (if (vctr? x)
    (alter! out! (fn ^double [^long i ^double cur]
                   (let [val (* (real/entry x i) (real/entry y i))]
                     (if reset? val (+ cur val)))))
    (alter! out! (fn ^double [^long i ^long j ^double cur]
                   (let [val (* (real/entry x i j) (real/entry y i j))]
                     (if reset? val (+ cur val)))))))

(defrecord ArgMaxTensorOp []
  cg/TensorOp
  (ensure-valid?! [this [X]]
    (when-not (= 1 (count (:shape X)))
      (throw (ex-info "Only handle vectors" {:X X})))
    true)
  (forward-node-pass! [this node]
    (let [output (:value node)
          X (-> node :children first :value)]
      (real/entry! output 0 (imax X))
      node))
  (backward-node-pass! [this node]
    (throw (ex-info "Not Supported"))))

(defrecord HadamardTensorOp []
  cg/TensorOp
  (ensure-valid?! [this [X Y]]
    (ensure-valid-shape?! (:shape X))
    (ensure-valid-shape?! (:shape Y))
    true)
  (forward-node-pass! [this node]
    (let [output (:value node)
          [X Y] (map :value (:children node))]
      (vect-math/mul! X Y output)
      node))
  (backward-node-pass! [this node]
    (let [dZ (:grad node)
          Z (:value node)
          [X Y] (map :value (:children node))
          [dX dY] (map :grad (:children node))
          tmp (when (or dX dY)
                (zero dZ))]
      (when dX
        (vect-math/mul! Y dZ tmp)
        (axpby! tmp dX))
      (when dY
        (vect-math/mul! X dZ tmp)
        (axpby! tmp dY)))
    node))

(defn dropout-mask [node]
  (let [t (-zeros (:shape node))
        ^double prob (get-in node [:graph-op :prob])]
    (alter! t (fn ^double [^double _]
                (if (< (Math/random) prob)
                  0.0
                  (/ 1.0 (- 1.0 prob)))))
    t))

(defrecord DropoutTensorOp []
  cg/TensorOp
  (ensure-valid?! [this inputs]
    (doseq [x inputs]
      (ensure-valid-shape?! (:shape x)))
    true)
  (forward-node-pass! [this node]
    (let [mask (dropout-mask node)
          input (first (:children node))]
      (vect-math/mul!
       mask
       (:value input)
       (:value node))
      (assoc node ::dropout-mask mask)))
  (backward-node-pass! [this node]
    (let [mask (::dropout-mask node)
          input (first (:children node))]
      (when-let [g (:grad input)]
        (let [tmp (zero g)]
          (vect-math/mul!
           mask
           (:grad node)
           tmp)
          (axpby! tmp g)))
      node)))

(defrecord SplitTensorOp []
  cg/TensorOp
  (ensure-valid?! [this inputs]
    (doseq [x inputs]
      (ensure-valid-shape?! (:shape x)))
    true)
  (forward-node-pass! [this node]
    (let [{:keys [dim, ^long start, ^long stop]} (:graph-op node)
          child (-> node :children first)
          v (:value child)]
      (case (count (:shape child))
        1 (copy! (subvector v start (- stop start)) (:value node))))
    node)
  (backward-node-pass! [this node]
    (let [{:keys [dim, ^long start, ^long stop]} (:graph-op node)
          child (-> node :children first)
          g (:grad node)]
      (when-let [cg (:grad child)]
        (case (count (:shape child))
          1 (transfer!
             g
             (subvector cg start (- stop start))))))))

(defrecord ConcatTensorOp []
  cg/TensorOp
  (ensure-valid?! [this inputs]
    (doseq [x inputs]
      (ensure-valid-shape?! (:shape x)))
    true)
  (forward-node-pass! [this node]
    (let [output (:value node)
          inputs (:children node)
          dim-to-cat (:dim-to-cat (:graph-op node))]
      (loop [inputs inputs offset 0]
        (when-let [input (first inputs)]
          (let [len (long (nth (:shape input) dim-to-cat))
                x (:value input)
                [nr nc] (:shape input)]
            (if (vctr? x)
              (copy! x (subvector output offset len))
              (if (= dim-to-cat 0)
                (copy! x (submatrix output offset 0 len nc))
                (copy! x (submatrix output 0 offset nr len))))
            (recur (next inputs) (+ offset len))))))
    node)
  (backward-node-pass! [this node]
    (let [output (:grad node)
          inputs (:children node)
          dim-to-cat (get-in node [:graph-op :dim-to-cat])]
      (loop [inputs inputs offset 0]
        (when-let [input (first inputs)]
          (let [len (long (nth (:shape input) dim-to-cat))
                dx (:grad input)
                [nr nc] (:shape input)]
            (when dx
              (if (vctr? dx)
                (np/axpby! (subvector output offset len) dx)
                (if (= dim-to-cat 0)
                  (np/axpby! (submatrix output offset 0 len nc) dx)
                  (np/axpby! (submatrix output 0 offset nr len) dx))))
            (recur (next inputs) (+ offset len)))))
      node)))

(defn element-wise-backward! [^IFn$DD dfx node]
  (let [dO (:grad node)]
    (when-let [dX (-> node :children first :grad)]
      (let [X (-> node :children first :value)]
        (if (vctr? dX)
          (alter! dX (fn ^double [^long i ^double cur]
                       (+ cur
                          (* (real/entry dO i)
                             (.invokePrim dfx (real/entry X i))))))
          (alter! dX (fn ^double [^long i ^long j ^double cur]
                       (+ cur
                          (* (real/entry dO i j)
                             (.invokePrim dfx (real/entry X i j)))))))))))

(defrecord ElementwiseTransformOp
  [^IFn$DD fx ^IFn$DD dfx]
  cg/TensorOp
  (ensure-valid?! [this [X]]
    (ensure-valid-shape?! (:shape X))
    true)
  (forward-node-pass! [this node]
    (let [output (:value node)
          X (-> node :children first :value)]
      (alter! output (fn ^double [^long i ^double _]
                       (.invokePrim fx (real/entry X i))))
      node))
  (backward-node-pass! [this node]
    (element-wise-backward! dfx node)))

(defrecord VecMathTransformOp
  [vec-fx ^IFn$DD dfx]
  cg/TensorOp
  (ensure-valid?! [this [X]]
    (ensure-valid-shape?! (:shape X))
    true)
  (forward-node-pass! [this node]
    (let [output (:value node)
          X (-> node :children first :value)]
      (vec-fx X output)
      node))
  (backward-node-pass! [this node]
    (element-wise-backward! dfx node)))

(let [s (Sigmoid.)]
  (defn ^:private ^:static sigmoid ^double  [^double x]
    (.value s x)))

(def ^:private +elementwise-op+
  ;; f(x) = e^x, df(x) = e^x
  {:exp [(fn -exp ^double [^double x] (FastMath/exp x))
         (fn -exp-d ^double [^double x] (FastMath/exp x))]
   ;; f(x) = 1/(1+e^{-x}, df(x) = (sigmoid(x)-1)/sigmoid(x)
   :sigmoid [(fn -sigmoid ^double [^double x]
               (sigmoid x))
             (fn -sigmoid-d ^double [^double x]
               (let [sig (sigmoid x)]
                 (* sig (- 1.0 sig))))]})

(def ^:private +vec-math-op+
  {;; f(x) = tanh(x), df(X) = 1 - tan(x)^2
   :tanh [vect-math/tanh!
          (fn -tanh-d ^double [^double x]
            (let [t (FastMath/tanh x)]
              (- 1.0 (* t t))))]})

(defn process-perf [perf-map]
  (let [total-ns (->> perf-map
                      (map (fn [e]
                         (let [{:keys [forward, backward]} (val e)]
                           (+ (.get ^AtomicLong forward)
                              (.get ^AtomicLong backward)))))
                      (reduce +))
        to-pct (fn [^AtomicLong l]
                 (/ (.get l) (double total-ns)))]
    (into {}
          (for [[k v] perf-map]
            [k (-> v
                   (update-in [:forward] to-pct)
                   (update-in [:backward] to-pct))]))))

(defn map-vals [f m]
  (into {} (for [e m] [(key e) (f (val e))])))

(def ^:private +tensor-ops+
  (merge
   {:+ ->SumTensorOp
    :* ->MultTensorOp
    :squeeze ->SqueezeTensorOp
    :sum-elems ->SumElemsTensorOp
    :strech ->StrechTensorOp
    :hadamard ->HadamardTensorOp
    :concat ->ConcatTensorOp
    :split ->SplitTensorOp
    :dropout ->DropoutTensorOp
    :cross-entropy-loss ->CrossEntropyLossTensorOp
    :arg-max ->ArgMaxTensorOp}
   (map-vals
    (fn [[fx dfx]] #(->VecMathTransformOp fx dfx))
    +vec-math-op+)
   (map-vals
    (fn [[fx dfx]] #(->ElementwiseTransformOp fx dfx))
    +elementwise-op+)))

(defn -from-nums [nums shape]
  (case (count shape)
    1 (dv nums)
    2 (let [[row col] shape]
        (-mk-matrix row col nums))
    ;; else
    (throw (ex-info "Unallowed shape" {:shape (vec shape)}))))

(defonce *cached-zeros (atom {}))

(defn zero-fill [factory t]
  (let [shape (flare/shape factory t)]
    (if-let [cached (get @*cached-zeros shape)]
      (copy! cached t)
      (let [z (flare/zeros factory shape)]
        (swap! *cached-zeros assoc shape z)
        (copy! z t)))))

(defn -transform-type [fn]
  (cond
    (number? fn) :double
    (instance? clojure.lang.IFn$DD fn) :dd-fn
    (instance? clojure.lang.IFn$DDD fn) :ddd-fn
    (instance? clojure.lang.IFn$ODD fn) :odd-fn
    (instance? clojure.lang.IFn$ODDD fn) :oddd-fn
    :else (throw (ex-info "Don't recognize fn"))))

(defrecord ^:private Factory []
  flare/PTensorFactory
  (get-op [this op-key]
    ((get +tensor-ops+ op-key)))
  (transform! [this tensor get-val]
    (let [get-val ^IFn$ODD get-val
          type (-transform-type get-val)
          dims (when (identical? type :odd-fn)
                 (long-array (if (vctr? tensor) 1 2)))
          fixed-return (double (if (identical? type :double) get-val 0.0))]
      (if (and (identical? type :double) (zero? fixed-return))
        (zero-fill this tensor)
        (alter! tensor
                (case [type (vctr? tensor)]
                  [:double true]
                  (fn cv ^double [^long i ^double x] fixed-return)
                  [:odd-fn true]
                  (fn ov ^double [^long i ^double x]
                    (aset dims (int 0) i)
                    (.invokePrim ^IFn$ODD get-val dims x))
                  [:dd-fn true]
                  (fn dv ^double [^long _ ^double x]
                    (.invokePrim ^IFn$DD get-val x))
                  [:double false]
                  (fn dm ^double [^long i ^long j ^double x] fixed-return)
                  [:odd-fn false]
                  (fn ff ^double [^long i ^long j ^double x]
                    (aset dims (int 0) i)
                    (aset dims (int 1) j)
                    (.invokePrim ^IFn$ODD get-val dims x))
                  [:dd-fn false]
                  (fn ff ^double [^long i ^long j ^double x]
                    (.invokePrim ^IFn$DD get-val x)))))))
  (transform! [this tensor other-tensor get-val]
    (let [get-val ^IFn$ODD get-val
          type (-transform-type get-val)
          dims (when (identical? type :od-fn)
                 (long-array (if (vctr? tensor) 1 2)))
          fixed-return (double (if (identical? type :double) 1.0 0.0))
          shape (flare/shape this tensor)
          oshape (flare/shape this other-tensor)]
     (when-not (= shape oshape)
       (throw (ex-info "Non-matching shapes"
                       {:shape shape :other-shape oshape})))
     (alter! tensor
             (case [type (vctr? tensor)]
               [:ddd-fn true]
               (fn tt ^double [^long i ^double x]
                 (let [other-val (real/entry other-tensor i)]
                   (.invokePrim ^IFn$DDD get-val x other-val)))
               [:oddd-fn true]
               (fn tf ^double [^long i  ^double x]
                 (aset dims (int 0) i)
                 (let [other-val (real/entry other-tensor i)]
                   (.invokePrim ^IFn$ODDD get-val dims x other-val)))
               [:ddd-fn false]
               (let [^IFn$DDD get-val get-val]
                 (fn ft ^double [^long i ^long j ^double x]
                   (let [other-val (real/entry other-tensor i j)]
                     (.invokePrim get-val x other-val))))
               [:oddd-fn false]
               (fn ff ^double [^long i ^long j ^double x]
                 (aset dims (int 0) i)
                 (aset dims (int 1) j)
                 (let [other-val (real/entry other-tensor i j)]
                   (.invokePrim ^IFn$ODDD get-val dims x other-val)))))))
  (shape [this t]
    (if (vctr? t)
      [(dim t)]
      [(mrows t) (ncols t)]))
  (from [this nums]
    (if (or (vctr? nums) (matrix? nums))
      nums
      (-from-nums nums (flare/guess-shape nums))))

  (copy! [this tensor! nums]
    (cond
      ;; fast if the nums is already neanderthal
      (or (matrix? nums) (vctr? nums)) (copy! nums tensor!)
      ;; faster to use nth and directly alter
      (vctr? tensor!) (let [nums (if (vector? nums) (vec nums) nums)]
                        (alter! tensor!
                                (fn ^double [^long idx ^double _]
                                  (nth nums idx))))
      ;; give up on performance....
      :else (let [shape (if (vctr? tensor!)
                          [(dim tensor!)]
                          [(mrows tensor!) (ncols tensor!)])]
              (copy! (-from-nums nums shape) tensor!))))
  (zeros [this shape]
    (-zeros shape))


  flare/-InternalPTensorFactory
  (debug-info [this]
    {:debug
     {:perf
      (->> (get-in (meta this) [:debug :perf])
           process-perf
           (sort-by (fn [e]
                      (let [{:keys [forward, backward]} (val e)]
                        (- (+ ^double forward ^double backward))))))}}))

(defn -build-perf-map []
  (into {}
        (for [k (keys +tensor-ops+)]
          [k {:forward (AtomicLong. 0)
              :backward (AtomicLong. 0)}])))

(defn factory [& [num-to-cache]]
  (let [f (->Factory)]
    (with-meta f (assoc-in (meta f) [:debug :perf] (-build-perf-map)))))
