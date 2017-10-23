(ns tensors.module
  (:refer-clojure :exclude [comp])
  (:require [tensors.node :as node]
            [tensors.model :as model]
            [tensors.computation-graph :as cg]
            [tensors.compute :as compute])
  (:import [tensors.node Node]))

(defprotocol Module
  (graph
    [this]
    [this input]
    [this input1 input2]
    "build a graph for any input(s)"))

(defprotocol ModelModule
  (-params [this] "map of unscoped parameter keys -> param-node"))

(defprotocol InputModule
  (-required-inputs [this]))

(defn forward! [factory module & inputs]
  (let [n (apply graph module inputs)]
    (compute/forward-pass! n factory)))

(defn predict [factory score-module & inputs]
  (when-let [n (apply graph score-module inputs)]
       (compute/forward-pass! (cg/arg-max n) factory)))

(defn params [module]
  (when (satisfies? ModelModule module)
    (-params module)))

(defn required-inputs [module]
  (when (satisfies? InputModule module)
    (-required-inputs module)))

(defn from-op [op]
  (when-not (satisfies? cg/GraphOp op)
    (throw (ex-info "Must be graph-op" {:op op})))
  (reify Module
    (graph [this input]
      (cg/add-graph-op op [input]))
    (graph [this input1 input2]
      (cg/add-graph-op op [input1 input2]))))

(defn comp [& ms]
  (reify
    Module
    (graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))

    ModelModule
    (-params [this] (merge params ms))

    InputModule
    (-required-inputs [this] (mapcat required-inputs ms))))


(defn affine
  "graph: returns single output (Wx + b)
   params: [W b]"
  [model num-out in-shape]
  (node/with-scope "affine"
    (let [[first-dim & rest] in-shape
          out-shape (vec (cons num-out rest))
          W (model/add-params! model [num-out first-dim] :name "W")
          b (model/add-params! model out-shape :name "b")]
      (reify Module
        (graph [this x]
          (cg/+ (cg/* W x) b))
        ModelModule
        (-params [this] [W b])))))
