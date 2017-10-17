(ns tensors.module
  (:refer-clojure :exclude [comp])
  (:require [tensors.node :as node]
            [tensors.model :as model]
            [tensors.computation-graph :as cg])
  (:import [tensors.node Node]))

(defprotocol Module
  (graph
    [this]
    [this input]
    [this input1 input2]))

(defprotocol ModelModule
  (-params [this])
  (predict-graph
    [this]
    [this input]
    [this input1 input2]))

(defprotocol InputModule
  (-required-inputs [this]))

(defn params [module]
  (when (satisfies? ModelModule module)
    (-params module)))

(defn required-inputs [module]
  (when (satisfies? InputModule module)
    (-required-inputs module)))

(defn from-op [op]
  (reify Module
    (graph [this input]
      (cg/add-graph-op
       op
       [input]))))

(defn comp [& ms]
  (reify
    Module
    (graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))

    ModelModule
    (-params [this] (mapcat params ms))
    (predict-graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))

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
