(ns tensors.module
  (:refer-clojure :exclude [comp])
  (:require [tensors.node :as node]
            [tensors.model :as model]
            [tensors.computation-graph :as cg])
  (:import [tensors.node Node]))

(defprotocol Module
  (params [this])
  (graph [this input]))

(defn from-op [op]
  (reify Module
    (params [this] [])
    (graph [this input]
      (cg/add-graph-op
       op
       [input]))))

(defn comp [& ms]
  (reify Module
    (params [this] (mapcat params ms))
    (graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))))


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
        (params [this] [W b])
        (graph [this x]
          (cg/+ (cg/* W x) b))))))
