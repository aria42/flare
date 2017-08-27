(ns tensors.graph
  (:refer-clojure :exclude [+ *])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [clojure.string :as str]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Schemas + Protocols

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
  (op-key [this])
  (forward-shape [this input-shapes]))

(s/defschema OpNode
  "Graph operation node schema"
  (assoc Node
         :graph-op GraphOp
         :children [(s/recursive #'Node)]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Making Graph Nodes

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
  {:type :op
   :shape (forward-shape op nodes)
   :graph-op op
   :ref-name (gensym (str (name (op-key op)) ":")) 
   :children nodes})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Graph Edge Operations

(defrecord SumGraphOp []
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


(defrecord MultGraphOp []
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
      (throw (ex-info (str prefix ": Must be row or column vector"))))))

(defrecord SqueezeGraphOp [dim-to-squeeze]
  GraphOp
  (op-key [this] :squeeze)
  (forward-shape [this inputs]
    (when-not (= 1 (count inputs))
      (throw (ex-info "Squeeze only takes single input"
                      {:causes inputs})))
    (let [shape (vec (:shape (first inputs)))]
      (when-not (= 1 (nth shape dim-to-squeeze))
        (throw (ex-info "Squeeze at ``dimension`` not 1"
                        {:dim-to-squeeze dim-to-squeeze
                         :shape shape})))
      (into (subvec shape 0 dim-to-squeeze)
            (subvec shape (inc dim-to-squeeze))))))

(defrecord StrechGraphOp [dim-to-insert]
  GraphOp
  (op-key [this] :strech)
  (forward-shape [this inputs]
    (when-not (= 1 (count inputs))
      (throw (ex-info "Stretch  only takes single input"
                      {:causes inputs})))
    (let [shape (vec (:shape (first inputs)))]
      (vec
       (concat (subvec shape 0 dim-to-insert)
               [1]
               (when (< (inc dim-to-insert) (count shape))
                 (subvec shape (inc dim-to-insert))))))))

(defrecord SoftMaxOp []
  GraphOp
  (op-key [this] :soft-max)
  (forward-shape [this input-nodes]
    (when (not= (count input-nodes) 1)
      (throw (ex-info "Exactly one input required")))
    (:shape (first input-nodes))))

(defrecord CrossEntropyLossOp []
  GraphOp
  (op-key [this] :cross-entropy-loss)
  (forward-shape [this [activations label :as input-nodes]]
    (when (or (nil? activations) (nil? label) (not= (count input-nodes) 2))
      (throw (ex-info "Expect (activations, labels) pair"
                      {:activations activations :label label})))
    (when-not (tensors/scalar-shape? (:shape label))
      (throw (ex-info "Label should be effectively scalar"
                      {:label-shape (:shape label)})))
    (when-not (tensors/vector-shape? (:shape activations))
      (throw (ex-info "Activations should be vector"
                      {:activations-shape (:shape activations)})))
    [1]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Public

(defn + [& inputs]
  (graph-edge (SumGraphOp.) inputs))

(defn * [& inputs]
  (graph-edge (MultGraphOp.) inputs))

(defn soft-max [input]
  (graph-edge (SoftMaxOp.) [input]))

(defn cross-entropy-loss [activations label]
  (graph-edge (CrossEntropyLossOp.) [activations label]))

(defn squeeze [input dim-to-squeeze]
  (graph-edge (SqueezeGraphOp. dim-to-squeeze) [input]))

(defn strech [input dim-to-insert]
  (graph-edge (StrechGraphOp. dim-to-insert) [input]))

(defmacro with-scope [^String scope-name & body]
  `(binding [*current-input-scope* (conj *current-input-scope* (name ~scope-name))]
     ~@body))

