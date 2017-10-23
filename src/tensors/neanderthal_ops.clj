(set! *unchecked-math* true)

(ns tensors.neanderthal-ops
  (:use [uncomplicate.neanderthal core native])
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [tensors.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as real]
            [schema.core :as s]
            [plumbing.core :as p]
            [tensors.cache-pool :as cache-pool])
  (:import [tensors.node Node]
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

(defrecord SumTensorOp []
  compute/TensorOp
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
  (prep [this node]
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
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [^Node n input-nodes]
      (ensure-valid-shape?! (.shape n)))
    (when-not (= 2 (count input-nodes))
      (throw (ex-info "Must have two arguments to MultTensorOp"))))
  (prep [this node]
    node)
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
          (np/mm! 1.0 (view-ge dZ) (trans (view-ge Y)) dX)))
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
    (let [[a b] (p/safe-get node :children)
          [nrows ncols] (:shape b)]
      ;; b is either vector or (row) vector
      (when (or (nil? ncols) (= ncols 1))
        [::mult (p/safe-get a :ref-name) (:shape b)])))
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
            (transfer! (p/safe-get b :value)
                       (submatrix new-input 0 j num-rows 1))
            (recur (inc j) (next nodes)))))
      ;; perfrom mm, copy columns to outputs
      (let [result (if cached (second cached)
                       (-mk-matrix (-> M :shape first) (count nodes)))]
        (scal! 0.0 result)
        (mm! 1.0 (p/safe-get M :value) new-input 1.0 result)
        (loop [nodes nodes cols (cols result)]
          (when-let [n (first nodes)]
            (transfer! (first cols) (p/safe-get n :value))
            (recur (next nodes) (next cols))))
        (if cached
          nodes
          (concat
           [(assoc (first nodes) :batched-cache [new-input result])]
           (rest nodes)))))))


(defrecord SqueezeTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (when-not (= 2 (count (.shape ^Node (first input-nodes))))
      (throw (ex-info "Need matrix shape"
                      {:shape (.shape ^Node (first input-nodes))}))))
  (prep [this node]
    node)
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
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (let [shape (:shape (first input-nodes))]
      (when-not (tensors/vector-shape? shape)
        (throw (ex-info "Need vector shape" {:shape shape})))))
  (prep [this node]
    node)
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
            (Math/exp (real/entry scores idx))))
  ;; normalize
  (let [Z (double (asum probs!))]
    (scal! (/ 1.0 Z) probs!)
    probs!))

(defrecord CrossEntropyLossTensorOp []
  compute/TensorOp
  (ensure-valid?! [this [activations label]]
    true)
  (prep [this node]
    (let [^Node node node
          [^Node activations-node ^Node label-node] (.children node)
          len (-> activations-node .shape first)]
      (assoc node ::probs (dv len))))
  (forward-node-pass! [this node]
    (let [^Node node node
          [^Node activations-node ^Node label-node] (.children node)
          activations (.value activations-node)
          probs (soft-max! activations (p/safe-get node ::probs))
          label (-> label-node .value (entry 0) long)
          correct-prob (real/entry probs label)]
      (when (>= label (dim probs))
        (throw (ex-info "Label index out-of-bounds"
                        {:label label
                         :dim (dim activations)})))
      (real/entry! (.value node) 0 (- (Math/log correct-prob)))
      node))
  (backward-node-pass! [this node]
    (let [^Node node node
          [^Node activations-node ^Node label-node] (.children node)
          probs (p/safe-get node ::probs)
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

(defn ^:static hadamard [out! x y reset?]
  (if (vctr? x)
    (alter! out! (fn ^double [^long i ^double cur]
                   (let [val (* (real/entry x i) (real/entry y i))]
                     (if reset? val (+ cur val)))))
    (alter! out! (fn ^double [^long i ^long j ^double cur]
                   (let [val (* (real/entry x i j) (real/entry y i j))]
                     (if reset? val (+ cur val)))))))

(defrecord ArgMaxTensorOp []
  compute/TensorOp
  (ensure-valid?! [this [X]]
    (when-not (= 1 (count (:shape X)))
      (throw (ex-info "Only handle vectors" {:X X})))
    true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [output (p/safe-get node :value)
          X (-> node :children first :value)]
      (real/entry! output 0 (imax X))
      node))
  (backward-node-pass! [this node]
    (throw (ex-info "Not Supported"))))

(defrecord HadamardTensorOp []
  compute/TensorOp
  (ensure-valid?! [this [X Y]]
    (ensure-valid-shape?! (:shape X))
    (ensure-valid-shape?! (:shape Y))
    true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [output (p/safe-get node :value)
          [X Y] (map :value (:children node))]
      (hadamard output X Y true)
      node))
  (backward-node-pass! [this node]
    (let [dZ (p/safe-get node :grad)
          Z (p/safe-get node :value)
          [X Y] (map :value (:children node))
          [dX dY] (map :grad (:children node))]
      (when dX
        (hadamard dX Y dZ false))
      (when dY
        (hadamard dY X dZ false)))
    node))

(defrecord ConcatTensorOp []
  compute/TensorOp
  (ensure-valid?! [this inputs]
    (doseq [x inputs]
      (ensure-valid-shape?! (:shape x)))
    true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [output (p/safe-get node :value)
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
    (let [output (p/safe-get node :grad)
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

(defrecord ElementwiseTransformOp
  [^IFn$DD fx ^IFn$DD dfx]
  compute/TensorOp
  (ensure-valid?! [this [X]]
    (ensure-valid-shape?! (:shape X))
    true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [output (p/safe-get node :value)
          X (-> node :children first :value)]
      (alter! output (fn ^double [^long i ^double _]
                       (.invokePrim fx (real/entry X i))))
      node))
  (backward-node-pass! [this node]
    (let [dO (p/safe-get node :grad)]
      (when-let [dX (-> node :children first :grad)]
        (let [X (p/safe-get node :value)]
          (if (vctr? dX)
            (alter! dX (fn ^double [^long i ^double cur]
                         (+ cur
                            (* (real/entry dO i)
                               (.invokePrim dfx (real/entry X i))))))
            (alter! dX (fn ^double [^long i ^long j ^double cur]
                         (+ cur
                            (* (real/entry dO i j)
                               (.invokePrim dfx (real/entry X i j)))))))))
      node)))

(defn ^:private ^:static sigmoid
  ^double [^double x]
  (/ 1.0 (+ 1.0 (Math/exp (- x)))))

(def ^:private +elementwise-op+
  ;; f(x) = e^x, df(x) = e^x
  {:exp [(fn -exp ^double [^double x] (Math/exp x))
         (fn -exp-d ^double [^double x] (Math/exp x))]
   ;; f(x) = 1/(1+e^{-x}, df(x) = (sigmoid(x)-1)/sigmoid(x)
   :sigmoid [(fn -sigmoid ^double [^double x]
               (sigmoid x))
             (fn -sigmoid-d ^double [^double x]
               (let [sig (sigmoid x)]
                 (* sig (- 1.0 sig))))]
   ;; f(x) = tanh(x), df(X) = 1 - tan(x)^2
   :tanh [(fn -tanh ^double [^double x]
            (Math/tanh x))
          (fn -tanh-d ^double [^double x]
            (let [t (Math/tanh x)]
              (- 1.0 (* t t))))]})

(def ^:private +tensor-ops+
  (merge
   {:+ ->SumTensorOp
    :* ->MultTensorOp
    :squeeze ->SqueezeTensorOp
    :strech ->StrechTensorOp
    :hadamard ->HadamardTensorOp
    :concat ->ConcatTensorOp
    :cross-entropy-loss ->CrossEntropyLossTensorOp
    :arg-max ->ArgMaxTensorOp}
   (p/map-vals
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
  (let [shape (tensors/shape factory t)]
    (if-let [cached (get @*cached-zeros shape)]
      (copy! cached t)
      (let [z (tensors/zeros factory shape)]
        (swap! *cached-zeros assoc shape z)
        (copy! z t)))))

(defrecord Factory []
  tensors/PFactory
  (get-op [this op-key]
    ((get +tensor-ops+ op-key)))
  (->clj [this tensor]
    (if (vctr? tensor)
      (seq tensor)
      (doall (map seq (rows tensor)))))
  (transform! [this tensor get-val]
    (let [constant? (number? get-val)
          fixed-return (double (if constant? get-val 0.0))
          dims (long-array (if (vctr? tensor) 1 2))
          get-val ^IFn$ODD get-val]
      (if (and constant? (zero? fixed-return))
        (zero-fill this tensor)
        (alter! tensor
                (case [constant? (vctr? tensor)]
                  [true true]
                  (fn tt ^double [^long i ^double x] fixed-return)
                  [false true]
                  (fn ft ^double [^long i ^double x]
                    (aset dims (int 0) i)
                    (.invokePrim get-val dims x))
                  [true false]
                  (fn tf ^double [^long i ^long j ^double x] fixed-return)
                  [false false]
                  (fn ff ^double [^long i ^long j ^double x]
                    (aset dims (int 0) i)
                    (aset dims (int 1) j)
                    (.invokePrim get-val dims x)))))))
  (transform! [this tensor other-tensor get-val]
    (let [ddd? (instance? IFn$DDD get-val)
          shape (tensors/shape this tensor)
          oshape (tensors/shape this other-tensor)
          ;; only need array when fn takes it
          dims (when-not ddd?
                 (long-array (if (vctr? tensor) 1 2)))]
     (when-not (= shape oshape)
       (throw (ex-info "Non-matching shapes"
                       {:shape shape :other-shape oshape})))
     (when (and (not ddd?) (not (instance? IFn$ODDD get-val)))
       (throw (ex-info "Bad get-val, must be IFn$DDD or IFn$ODDD primitive"
                       {:bad-fn get-val})))
     (alter! tensor
             (case [(vctr? tensor) ddd?]
               [true true]
               (fn tt ^double [^long i ^double x]
                 (let [other-val (real/entry other-tensor i)]
                   (.invokePrim ^IFn$DDD get-val x other-val)))
               [false true]
               (fn ft ^double [^long i ^long j ^double x]
                 (let [other-val (real/entry other-tensor i j)]
                   (.invokePrim ^IFn$DDD get-val x other-val)))
               [true false]
               (fn tf ^double [^long i  ^double x]
                 (aset dims (int 0) i)
                 (let [other-val (real/entry other-tensor i)]
                   (.invokePrim ^IFn$ODDD get-val dims x other-val)))
               [false false]
               (fn ff ^double [^long i ^long j ^double x]
                 (aset dims (int 0) i)
                 (aset dims (int 1) j)
                 (let [other-val (real/entry other-tensor i j)]
                   (.invokePrim ^IFn$ODDD get-val dims x other-val)))))))
  (shape [this t]
    (if (vctr? t)
      [(dim t)]
      [(mrows t) (ncols t)]))
  (from-nums [this nums]
    (if (or (vctr? nums) (matrix? nums))
      nums
      (-from-nums nums (tensors/guess-shape nums))))
  (grad-step! [this weights alpha grad]
    (axpy! (- (double alpha)) grad weights))

  (copy-from-input! [this tensor! nums]
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
    (case (count shape)
      1 (dv (first shape))
      2 (-mk-matrix (first shape) (second shape))
      (throw (ex-info "Unallowed shape for neanderthal"
                      {:shape (vec shape)})))))

(defn factory [& [num-to-cache]]
  (let [f (->Factory)]
    f
    #_(with-meta f
      {:cache (cache-pool/make
               (or num-to-cache 1)
               (fn [shape] (tensors/zeros f shape)))})))
