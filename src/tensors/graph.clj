(ns tensors.graph
  (:refer-clojure :exclude [+ *])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [clojure.string :as str]
            [tensors.graph :as graph]))

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
  (forward-shape [this input-shapes])
  (op-descriptor [this]))

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
;;; Graph Walks

(defn bottom-up-walk [node walk-fn]
  (walk-fn node (map #(bottom-up-walk % walk-fn) (:children node))))

(defn post-order-nodes [target]
  (conj (vec (mapcat post-order-nodes (:children target))) target))


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
      shape))
  (op-descriptor [this] "+"))


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
      [(ffirst shapes) (second (last shapes))]))
  (op-descriptor [this] "*"))

(s/defn ensure-vector-tensor?! [prefix shape :- tensors/Shape]
  (let [n (count shape)
        dim-counts (frequencies shape)]
    (when-not (and (= n 2) (= (get dim-counts 1) (dec n)))
      (throw (ex-info (str prefix ": Must be row or column vector"))))))

(defrecord SqueezeGraphOp [dim-to-squeeze]
  GraphOp
  (op-key [this] :squeeze)
  (op-descriptor [this] (str "squeeze-" dim-to-squeeze))
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
  (op-descriptor [this] (str "strech-" dim-to-insert))
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
  (op-descriptor [this] "soft-max")
  (forward-shape [this input-nodes]
    (when (not= (count input-nodes) 1)
      (throw (ex-info "Exactly one input required")))
    (:shape (first input-nodes))))

(defrecord CrossEntropyLossOp []
  GraphOp
  (op-key [this] :cross-entropy-loss)
  (op-descriptor [this] "cross-entropy-loss")
  (forward-shape [this [activations label :as input-nodes]]
    (when (or (nil? activations) (nil? label) (not= (count input-nodes) 2))
      (throw (ex-info "Expect (activations, labels) pair"
                      {:activations activations :label label})))
    (when-not (tensors/scalar-shape? (:shape label))
      (throw (ex-info "Label sjhould be effectively scalar"
                      {:label-shape (:shape label)})))
    (when-not (tensors/vector-shape? (:shape activations))
      (throw (ex-info "Activations should be vector"
                      {:activations-shape (:shape activations)})))
    [1]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Display/Summarize Graphs


(s/defn ^:private display-name [node]
  (case (:type node)
    :input (format "input(%s, %s)" (:ref-name node) (:shape node))
    ;; else
    (:ref-name node)))

(s/defn generate-equations :- [s/Str]
  [target :- Node]
  (for [n (post-order-nodes target) :when (= (:type n) :op)]
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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Public Graph Operations


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
  `(binding [*current-input-scope*
             (conj *current-input-scope* (name ~scope-name))]
     ~@body))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Sample Graphs

(def lr
  (let [num-classes 2
        num-feats 3
        W (graph/input "W" [num-classes num-feats])
        b (graph/strech (graph/input "bias" [num-classes]) 1)
        feat-vec (graph/strech (graph/input "f" [num-feats]) 1)
        activations (graph/squeeze (graph/+ (graph/* W feat-vec) b) 1)
        probs (graph/soft-max activations)
        label (graph/input "label" [1])
        loss (graph/cross-entropy-loss probs label)]
    {:loss loss
     :activations activations}))


(def simple-graph
  (let [X (graph/input "X" [2 2])
        Y (graph/input "Y" [2 2])
        Z (graph/input "Z" [2 2])]
    (graph/* Z (graph/+ X Y))))


