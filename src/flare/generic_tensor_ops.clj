(ns flare.generic-tensor-ops
  (:require [flare.computation-graph :as cg]
            [flare.core :as flare]))

(defrecord ScaleTensorOp []
  cg/TensorOp
  (ensure-valid?! [this nodes])
  (forward-node-pass! [this node]
    (let [child-val (-> node :children first :value)
          scale (-> node :graph-op :scalar)]
      (flare/copy! (:value node) child-val)
      (flare/scale! (:value node) scale))
    node)
  (backward-node-pass! [this node]
    (let [child-grad (-> node :children first :grad)
          scale (-> node :graph-op :scalar)]
      (when child-grad
        (flare/copy! child-grad (:grad node))
        (flare/scale! child-grad scale)))))


(defn tensor-ops []
  {:scale ->ScaleTensorOp})
