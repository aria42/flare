(ns flare.computation-graph
  "Abstractions for constructing computation graphs using graph operations.
   The public methods for operations are at bottom of this ns.
   The key protocols are  as follows:

   * `GraphOp` a computational graph operation that knows nothing about tensor
     implementation, but can validate if inputs nodes are valid and predict
     output shape
   * `TensorOp` an implementation of a specific graph operation which performs
     forward/backwad operation.
  "
  (:refer-clojure :exclude [+ * concat max])
  (:require [flare.core :as flare]
            [flare.graph :as graph]
            [clojure.string :as str]
            [flare.node :as node]
            [flare.model :as model]
            [flare.graph :as graph]
            [flare.cache-pool :as cache-pool])
  (:import [clojure.lang Keyword]
           [flare.node Node]
           [java.util.concurrent.atomic AtomicLong]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Protocols

(defprotocol GraphOp
  "Graph operation only needs to be aware of shape of output,
   independent of any tensor implementation."
  (op-key [this]
    "A keyword that uniquely identifies the graph operation,
    useful for connecting to a tensor implementation.")
  (op-validate! [this nodes]
    "throws exception if the operation isn't valid for the argument nodes")
  (forward-shape [this input-shapes]
    "returns the shape of the operation")
  (op-descriptor [this]
    "returns a text description of the operation, useful for generating
     equations of the graph computations, can include the shape
     or other arguments"))

(defprotocol TensorOp
  "A tensor op executes a given `GraphOp` in a tensor implementation"
  (ensure-valid?! [this input-nodes]
    "Ensure the operation can be perfed with the tensor operation. Some
    impls may support limited dimension or sizes")
  (forward-node-pass! [this node]
    "compute the forward pass of the algorithm, for each node, compute
     `:value` tensor for passed in node, using the `:children` nodes
     and their `:value` tensors. Returns the node in case any other
     computations are added to the node for use in the backward pass.")
  (backward-node-pass! [this node]
    "compute the `:grad` gradient tensor on each child of passed in node reaching
     down to the leaves  (which include the parameter nodes).
     Returns the node so that downstream backward-node-pass!
     calls can use added data."))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Adding Graph Op

(defn -tensor-op
  "valdiates that tensor op valid for a computation,
   and returns `TensorOp`"
  [^Node node factory]
  (assert (identical? :op (.type node)))
  (let [op-key (-> node .graph-op op-key)
        tensor-op (flare/get-op factory op-key)]
    (ensure-valid?! tensor-op (.children node))
    tensor-op))

(defn -tensor [^Node node key factory cache]
  (if-let [t (get node key)]
    t
    (if cache
      (cache-pool/get-obj cache (.shape node))
      (flare/zeros factory (.shape node)))))

(defn -forward
  ([^Node node factory cache]
   (let [tensor-op (-tensor-op node factory)
         ok (-> node .graph-op op-key)
         node (assoc node
                     :value (-tensor node :value factory cache)
                     :grad (-tensor node :grad factory cache))
         perf-map (-> factory meta (get-in [:debug :perf]))
         start (System/nanoTime)
         node (forward-node-pass! tensor-op  node)
         end (System/nanoTime)
         sum-nanos (get-in perf-map [ok :forward])]
     (when sum-nanos
       (.getAndAdd ^AtomicLong sum-nanos (- end start)))
     node)))

(defn add-graph-op
  "create a new node using a `GraphOp` and a sequence of nodes. If
   in `eager?` mode, will execute the tensor operation as well and
   result will be in `:tensor` field on returned node

   The `nodes` inputs will be on the `:children` field of returned `Node`"
  [op nodes]
  (op-validate! op nodes)
  ;; Bottleneck so using java constructor
  (let [shape (forward-shape op nodes)
        node-name (node/scoped-name (node/gen-name (name (op-key op))))
        node (Node. :op shape node-name nil nil op nodes)
        {:keys [eager?, factory, cache]} (flare/state)]
    (if eager?
      (-forward node factory cache)
      node)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Display/Summarize Graphs

(defn ^:private display-name [node]
  (if (identical? (:type node) :input)
    (format "input(%s, %s)" (:ref-name node) (:shape node))
    ;; else
    (:ref-name node)))

(defn generate-equations
  "generate semi-readable equations for computations in DAG
   represented by `target`"
  [^Node target]
  (for [n (graph/post-order-nodes target) :when (= (:type n) :op)]
    (format "%s = (%s %s) ;; shape: %s"
            (:ref-name n)
            (-> n :graph-op op-descriptor)
            (str/join " " (map display-name (:children n)))
            (:shape n))))

(defn summarize-computation
  "Create informative s-expression for computation"
  ([^Node target ^long indent]
   (str
    (when (> indent 0)
      (str "\n" (str/join (repeat indent "  "))))
    (case (:type target)
      :op
      (format "(%s :shape %s %s)"
              (-> target :graph-op op-descriptor)
              (:shape target)
              (str/join " " (map #(summarize-computation % (inc indent))
                                 (:children target))))
      :input
      (format "input(%s, %s)" (:ref-name target) (:shape target))
      :params
      (format "param(%s, %s)" (:ref-name target) (:shape target))
      :constant
      (format "constant(%s, %s) " (:ref-name target) (:shape target))
      ;; else
      (throw (ex-info "Bad node to summarize" {:node target})))))
  ([target] (summarize-computation target 0)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Graph Edge Operations

(defrecord SumGraphOp []
  GraphOp
  (op-key [this] :+)
  (op-validate! [this input-nodes]
    (when (empty? input-nodes)
      (throw (ex-info "Has empty inputs" {:inputs []})))
    (let [[shape & the-rest :as shapes] (map :shape input-nodes)]
      (when-not (every? #(= shape %) the-rest)
        (throw (ex-info "Not all input shpaes match"
                        {:shapes (vec shapes)})))))
  (forward-shape [this input-nodes]
    (:shape (first input-nodes)))
  (op-descriptor [this] "+"))

(defn vec-remove
  "remove elem in coll"
  [coll ^long pos]
  (vec (clojure.core/concat (subvec coll 0 pos) (subvec coll (inc pos)))))

(defrecord MultGraphOp []
  GraphOp
  (op-key [this] :*)
  (op-validate! [this input-nodes]
    (when (empty? input-nodes)
      (throw (ex-info "Need at least one input"
                      {:num-args (count input-nodes)})))
    (let [shapes (map :shape input-nodes)]
      (when-let [bad (some (fn [s] (or (nil? s) (> (count s) 2))) shapes)]
        (throw (ex-info "Only max 2d allowed" {:shapes shapes})))
      (when-not (every?
                 (fn [[[a b] [c d]]] (= b c))
                 (partition 2 shapes))
        (throw (ex-info "Incompatibl shapes" {:shapes shapes})))))
  (forward-shape [this input-nodes]
    (let [shapes (map :shape input-nodes)
          num-rows (ffirst shapes)]
      (if-let [num-cols (second (last shapes))]
        [num-rows num-cols]
        [num-rows])))
  (op-descriptor [this] "*"))

(defrecord ArgMaxGraphOp []
  GraphOp
  (op-key [this] :arg-max)
  (op-validate! [this nodes]
    (when-not (= 1 (count nodes))
      (throw (ex-info "Required 1 arg" {:nodes nodes}))))
  (op-descriptor [this] "arg-max")
  (forward-shape [this nodes] [1]))

(defn ensure-vector-tensor?! [prefix shape]
  (let [n (count shape)
        dim-counts (frequencies shape)]
    (when-not (and (= n 2) (= (get dim-counts 1) (dec n)))
      (throw (ex-info (str prefix ": Must be row or column vector")
                      {:dim-counts dim-counts})))))

(defrecord SqueezeGraphOp [dim-to-squeeze]
  GraphOp
  (op-key [this] :squeeze)
  (op-descriptor [this] (str "squeeze-" dim-to-squeeze))
  (op-validate! [this nodes]
    (when-not (= 1 (count nodes))
      (throw (ex-info "Squeeze only takes single input"
                      {:num-args (count nodes)})))
    (let [shape (vec (:shape (first nodes)))]
      (when-not (= 1 (nth shape dim-to-squeeze))
        (throw (ex-info "Squeeze at ``dimension`` not 1"
                        {:dim-to-squeeze dim-to-squeeze
                         :shape shape})))
      (vec-remove shape dim-to-squeeze)))
  (forward-shape [this inputs]
    (let [shape (vec (:shape (first inputs)))]
      (vec-remove shape dim-to-squeeze))))

(defrecord SumElemsGraphOp []
  GraphOp
  (op-key [this] :sum-elems)
  (op-descriptor [this] "sum-elems")
  (op-validate! [this nodes]
    (when-not (= 1 (count nodes))
      (throw (ex-info "SumElems only takes single input"
                      {:num-args (count nodes)}))))
  (forward-shape [this inputs] [1]))

(defrecord StrechGraphOp [^long dim-to-insert]
  GraphOp
  (op-key [this] :strech)
  (op-descriptor [this] (str "strech-" dim-to-insert))
  (op-validate! [this inputs]
    (when-not (= 1 (count inputs))
      (throw (ex-info "Stretch  only takes single input"
                      {:causes inputs}))))
  (forward-shape [this inputs]
    (let [shape (vec (:shape (first inputs)))]
      (vec
       (clojure.core/concat (subvec shape 0 dim-to-insert)
               [1]
               (when (< (inc dim-to-insert) (count shape))
                 (subvec shape (inc dim-to-insert))))))))

(defrecord CrossEntropyLossOp []
  GraphOp
  (op-key [this] :cross-entropy-loss)
  (op-descriptor [this] "cross-entropy-loss")
  (op-validate! [this [activations label & other]]
    (when (or (nil? activations) (nil? label) (seq other))
      (throw (ex-info "Expect (activations, labels) pair"
                      {:activations activations :label label})))
    (when-not (flare/scalar-shape? (:shape label))
      (throw (ex-info "Label sjhould be effectively scalar"
                      {:label-shape (:shape label)})))
    (when-not (flare/vector-shape? (:shape activations))
      (throw (ex-info "Activations should be vector"
                      {:activations-shape (:shape activations)}))))
  (forward-shape [this [activations label :as input-nodes]]
    [1]))

(defrecord HadamardProduct []
  GraphOp
  (op-key [this] :hadamard)
  (op-descriptor [this] "hadamard")
  (op-validate! [this [X Y & inputs]]
    (when (or (nil? X) (nil? Y))
      (throw (ex-info "Invalid inputs, must have 2" {:inputs inputs})))
    (when-not (= (:shape X) (:shape Y))
      (throw (ex-info "Need equal shapes" {:shapes (map :shape [X Y])}))))
  (forward-shape [this [X Y]]
    ;; output has same shape
    (:shape X)))

(defrecord ConcatOp [dim-to-cat]
  GraphOp
  (op-key [this] :concat)
  (op-descriptor [this] (str "concat-" dim-to-cat))
  (op-validate! [this inputs]
    (when (empty? inputs)
      (throw (ex-info "Empty inputs")))
    (let [shapes (map (comp vec :shape) inputs)
          dropped-shapes (map #(vec-remove % dim-to-cat) shapes)]
      (when-let [bad (some #(not= % (first dropped-shapes)) dropped-shapes)]
        (throw (ex-info "Non-matching shpapes" {:bad shapes})))))
  (forward-shape [this inputs]
    (let [shapes (map (comp vec :shape) inputs)
          concat-len (reduce clojure.core/+ (map #(nth % dim-to-cat) shapes))]
      (assoc (first shapes) dim-to-cat concat-len))))

(defrecord SplitOp [^long dim ^long start ^long stop]
  GraphOp
  (op-key [this] :split)
  (op-descriptor [this] (format "split(%d, %d, %d)" dim start stop))
  (op-validate! [this inputs]
    (when-not (= (count inputs) 1)
      (throw (ex-info "Must have exactly one arg")))
    (when (<= stop start)
      (throw (ex-info "stop must be > start"
                      {:stop stop :start start})))
    (let [n (first inputs)
          ^long dim-len (nth (:shape n) dim)]
      (when (or (< start 0) (> stop dim-len))
        (throw (ex-info "Out of dimension"
                        {:start start, :stop stop :dim-len dim-len})))))
  (forward-shape [this [node]]
    (let [shape (:shape node)]
      (vec
       (clojure.core/concat
        (subvec shape 0 dim)
        [(- stop start)]
        (when (< (inc dim) (count shape))
          (subvec shape (inc dim))))))))

(defrecord MaxOp []
  GraphOp
  (op-key [this] :max)
  (op-descriptor [this] "max")
  (op-validate! [this [x & inputs]]
    (when-not (every? #(= (:shape x) (:shape %)) inputs)
      (throw (ex-info "All inputs must have same shape"
                      {:shapes (map :shape inputs)}))))
  (forward-shape [this inputs]
    (-> inputs first :shape)))

(defrecord DropoutGraphOp [^double prob]
  GraphOp
  (op-key [this] :dropout)
  (op-validate! [this [input-node :as ns]]
    (when-not  (= (count ns) 1)
      (throw (ex-info "Require exactly arg"
                      {:num-args (count ns)}))))
  (forward-shape [this [input-node]]
    (:shape input-node))
  (op-descriptor [this] (format "dropout(%s)" prob)))

(defn scalar-op
  "graph operation of single argument where output is same shape"
  [key]
  (reify GraphOp
    (op-key [this] key)
    (op-descriptor [this] (name key))
    (op-validate! [this [X]]
      (when (nil? X)
        (throw (ex-info "Must have input" {:X X}))))
    (forward-shape [this [X]]
      ;; output has same shape
      (:shape X))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Public Atomic Graph Operations


(defn + [& inputs]
  (add-graph-op (SumGraphOp.) inputs))

(defn * [& inputs]
  (add-graph-op (MultGraphOp.) inputs))

(defn cross-entropy-loss [activations label]
  (add-graph-op (CrossEntropyLossOp.) [activations label]))

(defn squeeze [input dim-to-squeeze]
  (add-graph-op (SqueezeGraphOp. dim-to-squeeze) [input]))

(defn strech [input dim-to-insert]
  (add-graph-op (->StrechGraphOp dim-to-insert) [input]))

(defn exp [input]
  (add-graph-op (scalar-op :exp) [input]))

(defn sigmoid [input]
  (add-graph-op (scalar-op :sigmoid) [input]))

(defn tanh [input]
  (add-graph-op (scalar-op :tanh) [input]))

(defn dropout [p input]
  (add-graph-op (->DropoutGraphOp p) [input]))

(defn hadamard
  "Output is element-wise product of two inputs"
  [x y]
  (add-graph-op (HadamardProduct.) [x y]))

(defn arg-max
  [x]
  (add-graph-op (ArgMaxGraphOp.) [x]))

(defn concat
  [dim & inputs]
  ;; handle no-op for singleton input
  (if (= (count inputs) 1)
    (first inputs)
    (add-graph-op (->ConcatOp dim) inputs)))

(defn max
  "Inputs: Tensors of same shape
   Output: Tensor of input shape where each element is maximum"
  [& inputs]
  (add-graph-op (->MaxOp) inputs))

(defn split
  [node dim & split-indices]
  (let [dim-len (nth (:shape node) dim)
        splits (clojure.core/concat [0] split-indices [dim-len])]
    (map (fn [[start stop]]
           (add-graph-op (->SplitOp dim start stop) [node]))
         (butlast (partition-all 2 1 splits)))))

(defn sum [node]
  (add-graph-op (SumGraphOp.) [node]))
