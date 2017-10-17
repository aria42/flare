(ns tensors.computation-graph
  (:refer-clojure :exclude [+ * concat])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [clojure.string :as str]
            [tensors.node :as node]
            [tensors.model :as model]
            [tensors.graph :as graph]
            [plumbing.core :as p])
  (:import [clojure.lang Keyword]
           [tensors.node Node]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Schemas + Protocols

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
     equations of the graph computations"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Adding Graph Op


(s/defn add-graph-op
  [op :- GraphOp  nodes]
  (op-validate! op nodes)
  ;; Bottleneck so using java constructor
  (Node.
   :op
   (forward-shape op nodes)
   (node/scoped-name (node/gen-name (name (op-key op))))
   ;; value
   nil
   ;; grad
   nil
   op
   ;; tensor-op
   nil
   ;; children
   nodes))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Display/Summarize Graphs

(s/defn ^:private display-name [node]
  (case (:type node)
    :input (format "input(%s, %s)" (:ref-name node) (:shape node))
    ;; else
    (:ref-name node)))

(s/defn generate-equations :- [s/Str]
  [target]
  (for [n (graph/post-order-nodes target) :when (= (:type n) :op)]
    (format "%s = (%s %s) ;; shape: %s"
            (:ref-name n)
            (-> n :graph-op op-descriptor)
            (str/join " " (map display-name (:children n)))
            (:shape n))))

(s/defn summarize-computation
  "Create informative s-expression for computation"  
  ([target indent :- s/Int]
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
  [coll pos]
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
        (throw (ex-info "Only max 2d allowed" {:bad bad})))
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

(s/defn ensure-vector-tensor?! [prefix shape :- tensors/Shape]
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

(defrecord StrechGraphOp [dim-to-insert]
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
    (when-not (tensors/scalar-shape? (:shape label))
      (throw (ex-info "Label sjhould be effectively scalar"
                      {:label-shape (:shape label)})))
    (when-not (tensors/vector-shape? (:shape activations))
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

(defn hadamard
  "Output is element-wise product of two inputs"
  [x y]
  (add-graph-op (HadamardProduct.) [x y]))

(defn arg-max
  [x]
  (add-graph-op (ArgMaxGraphOp.) [x]))

(defn concat
  [dim & inputs]
  (add-graph-op (->ConcatOp dim) inputs))
