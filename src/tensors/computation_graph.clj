(ns tensors.computation-graph
  (:refer-clojure :exclude [+ *])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [clojure.string :as str]
            [tensors.node :as node]
            [tensors.graph :as graph])
  (:import [clojure.lang Keyword]))

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
;;;  Making Graph Nodes


(def ^:dynamic *current-input-scope* [])

(defn ^String full-node-name
  "fully qualified node name with scopes"
  [^String node-name]
  (str/join "/" (conj *current-input-scope* node-name)))

(defmacro with-scope [^String scope-name & body]
  `(binding [*current-input-scope*
             (conj *current-input-scope* (name ~scope-name))]
     ~@body))

(s/defn input
  "Create input variable node, using provided input-name
   or generating one if one isn't provided"
  ([input-name :- String shape :- tensors/Shape]
   (node/map->Node
    {:type :input
     :shape shape
     :ref-name (full-node-name input-name)}))
  ([shape :- tensors/Shape]
   (input (name (gensym "input")) shape)))

(s/defn constant
  "Create input variable node, using provided input-name
   or generating one if one isn't provided"
  [input-name :- String shape :- tensors/Shape value :- s/Any]
  (node/map->Node
   {:type :constant
    :shape shape
    :value value
    :ref-name (full-node-name input-name)})
  ([shape :- tensors/Shape value :- s/Any]
   (constant (name (gensym "constant")) shape value)))

(defmacro definput [input-var shape]
  `(def ~input-var (input ~(name input-var) ~shape)))

(defmacro defparams [params-var shape]
  `(def ~params-var (params ~(name params-var) ~shape)))

(s/defn add-graph-op
  [op :- GraphOp  nodes]
  (op-validate! op nodes)
  (node/map->Node
   {:type :op
    :shape (forward-shape op nodes)
    :graph-op op
    :ref-name (full-node-name (gensym (str (name (op-key op)) ":")))
    :children nodes}))

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
      ;; else
      (throw (ex-info "Bad node to summarize" {:node target})))))
  ([target] (summarize-computation target 0)))
