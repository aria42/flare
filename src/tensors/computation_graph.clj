(ns tensors.computation-graph
  (:refer-clojure :exclude [+ *])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [clojure.string :as str]
            [tensors.graph :as graph]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Schemas + Protocols

(s/defschema Node
  "Generic graph node schema"
  {:type s/Keyword
   :shape tensors/Shape
   :ref-name s/Str
   ;; other keys are allowed
   s/Any s/Any})

(s/defschema InputNode
  "Input graph node schema"
  (assoc Node :type (s/eq :input)))

(s/defschema InitParamSpec
  "Spec for how to generate parameter entries independently
  implement `get-param-rng` multi-method for `:distribution`
  for a new distirbution"
  {:distribution s/Keyword
   s/Any s/Any})

(defmulti ^clojure.lang.IFn$D get-param-rng :distribution)

(s/defschema ParamsNode
  "Params graph node schema"
  (assoc Node
         :init InitParamSpec
         :type (s/eq :params)))

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

(s/defschema OpNode
  "Graph operation node schema"
  (assoc Node
         :type (s/eq :op)
         :graph-op GraphOp
         :children [Node]))

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

(s/defn input :- InputNode
  "Create input variable node, using provided input-name
   or generating one if one isn't provided"
  ([input-name :- String shape :- tensors/Shape]
   {:type :input
    :shape shape
    :ref-name (full-node-name input-name)})
  ([shape :- tensors/Shape]
   (input (name (gensym "input")) shape)))


(s/defn params :- ParamsNode
  "Create params variable node. See \"params.clj\" for how we init"
  ([shape :- tensors/Shape]
   (params (name (gensym "params")) shape))
  ([param-name :- String shape :- tensors/Shape]
   (params param-name shape {:distribution :uniform :upper 1.0 :lower -1.0}))
  ([param-name :- String shape :- tensors/Shape init-spec :- InitParamSpec]
   {:type :params
    :shape shape
    :init init-spec
    :ref-name (full-node-name param-name)}))

(defmacro definput [input-var shape]
  `(def ~input-var (input ~(name input-var) ~shape)))

(defmacro defparams [params-var shape]
  `(def ~params-var (params ~(name params-var) ~shape)))

(s/defn add-graph-op :- OpNode
  [op :- GraphOp  nodes :- [Node]]
  (op-validate! op nodes)
  {:type :op
   :shape (forward-shape op nodes)
   :graph-op op
   :ref-name (gensym (str (name (op-key op)) ":")) 
   :children nodes})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Display/Summarize Graphs

(s/defn ^:private display-name [node]
  (case (:type node)
    :input (format "input(%s, %s)" (:ref-name node) (:shape node))
    ;; else
    (:ref-name node)))

(s/defn generate-equations :- [s/Str]
  [target :- Node]
  (for [n (graph/post-order-nodes target) :when (= (:type n) :op)]
    (format "%s = (%s %s) ;; shape: %s"
            (:ref-name n)
            (-> n :graph-op op-descriptor)
            (str/join " " (map display-name (:children n)))
            (:shape n))))

(s/defn summarize-computation
  "Create informative s-expression for computation"
  ([target :- Node] (summarize-computation target 0))
  ([target :- Node indent :- s/Int]
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
      ;; else
      (throw (ex-info "Bad node to summarize" {:node target}))))))
