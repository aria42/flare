(ns tensors.graph
  (:refer-clojure :exclude [+ *])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [clojure.string :as str]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  Schemas + Protocols
;;;

(s/defschema Node
  "Generic graph node schema"
  {:type s/Keyword
   :shape tensors/Shape
   :ref-name s/Str})

(s/defschema InputNode
  "Input graph node schema"
  (assoc Node :type :input))

(s/defschema ParamsNode
  "Params graph node schema"
  (assoc Node :type :params))

(defprotocol GraphOp
  "Graph operation only needs to be aware of shape of output,
   independent of any tensor implementation."
  (^Keyword op-key [this])
  (^Shape forward-shape [this input-shapes]))

(s/defschema OpNode
  "Graph operation node schema"
  (assoc Node
         :graph-op GraphOp
         :children [(s/recursive #'Node)]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  Making Graph Nodes
;;;

(def ^:dynamic *current-input-scope* [])

(defn ^String full-node-name
  "fully qualified node name with scopes"
  [^String node-name]
  (str/join "/" (conj *current-input-scope* node-name)))

(s/defn input :- InputNode
  "Create input variable node"
  ([input-name :- String shape :- tensors/Shape]
   {:type :input
    :shape shape
    :ref-name (full-node-name input-name)})
  ([shape :- tensors/Shape]
   (input (name (gensym "input")) shape)))


(s/defn params :- ParamsNode
  "Create params variable node"
  [param-name :- String shape :- tensors/Shape]
  {:type :params
   :shape shape
   :ref-name (full-node-name param-name)})

(defmacro definput [input-var shape]
  `(def ~input-var (input ~(name input-var) ~shape)))

(defmacro defparams [params-var shape]
  `(def ~params-var (params ~(name params-var) ~shape)))

(s/defn ^:private graph-edge :- OpNode
  [op :- GraphOp  nodes :- [Node]]
  (try
    {:type :op
     :shape (forward-shape op nodes)
     :graph-op op
     :ref-name (gensym (str (name (op-key op)) ":")) 
     :children nodes}
    (catch RuntimeException e
      (let [msg (format "Operation issue '%s' on shapes %s: %s"
                        (-> op op-key name)
                        (str/join "," (map :shape nodes))
                        (.getMessage e))]
        (throw (RuntimeException. msg))))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  Graph Edge Operations
;;;

(deftype SumGraphOp []
  GraphOp
  (op-key [this] :+)
  (forward-shape [this input-nodes]
    (when (empty? input-nodes)
      (throw (RuntimeException. "Has empty inputs")))
    (let [[shape & the-rest :as shapes] (map :shape input-nodes)]
      (when-not (every? #(= shape %) the-rest)
        (throw (RuntimeException.
                (format "Not all input shapes match: %s" (vec shapes)))))
      shape)))


(deftype MultGraphOp []
  GraphOp
  (op-key [this] :*)
  (forward-shape [this input-nodes]
    (when (empty? input-nodes)
      (throw (RuntimeException. "Need at least one input")))
    (let [shapes (map :shape input-nodes)]
      (when-not (every?
                 (fn [[[a b] [c d]]]
                   (= b c))
                 (partition 2 shapes))
        (throw (RuntimeException.
                (format "Can't multiply shapes: %s" shapes))))
      [(ffirst shapes) (second (last shapes))])))

(s/defn ensure-vector-tensor?! [prefix shape :- tensors/Shape]
  (let [n (count shape)
        dim-counts (frequencies shape)]
    (when-not (and (= n 2) (= (get dim-counts 1) (dec n)))
      (throw (RuntimeException. (str prefix ": Must be row or column vector"))))))

(deftype SoftMaxOp []
  GraphOp
  (op-key [this] :soft-max)
  (forward-shape [this input-nodes]
    (when (not= (count input-nodes) 1)
      (throw (RuntimeException. "Exactly one input required")))
    (let [first-shape (-> input-nodes first :shape)]
      (ensure-vector-tensor?! "input" first-shape)
      first-shape)))

(deftype CrossEntropyLossOp []
  GraphOp
  (op-key [this] :cross-entropy-loss)
  (forward-shape [this [activations label :as input-nodes]]
    (when (or (nil? activations) (nil? label) (not= (count input-nodes) 2))
      (throw (RuntimeException. "Expect (activations, labels) pair")))
    (when-not (= [1] (:shape label))
      (throw (RuntimeException. "Label should be a [1] shape tensor")))
    (let [act-shape (:shape activations)]
      (ensure-vector-tensor?! "activations" act-shape)
      [1])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  Public
;;;

(defn + [& inputs]
  (graph-edge (SumGraphOp.) inputs))

(defn * [& inputs]
  (graph-edge (MultGraphOp.) inputs))

(defn soft-max [input]
  (graph-edge (SoftMaxOp.) [input]))

(defn cross-entropy-loss [activations label]
  (graph-edge (CrossEntropyLossOp.) [activations label]))

(defmacro with-scope [^String scope-name & body]
  `(binding [*current-input-scope* (conj *current-input-scope* (name ~scope-name))]
     ~@body))

(def +core-ops+
  {:+ (SumGraphOp.)
   :* (MultGraphOp.)
   :cross-entropy-loss (CrossEntropyLossOp.)
   :soft-max  (SoftMaxOp.)})
