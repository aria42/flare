(ns flare.batch-bench-test
  (:use [uncomplicate.neanderthal core native]
        [tensors neanderthal-ops])
  (:require [flare.graph :as graph]
            [flare.core :as flare]
            [tensors.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as real]
            [schema.core :as s]
            [plumbing.core :as p]))

(defn gen-data [n]
  (let [num-classes 10
        num-feats 500
        M (dge num-classes num-feats)
        r (java.util.Random. 0)
        vecs (mapv (fn [_] (dv num-feats)) (range n))]
    (alter! M (fn ^double [^double x] (.nextDouble ^java.util.Random r)))
    (doseq [v vecs]
      (alter! v (fn ^double [^double x] (if (.nextBoolean r) 1.0 0.0))))
    (let [param {:ref-name "M" :shape [num-classes num-feats] :value M}]
      (mapv (fn [v]
              {:shape [num-classes]
               :value (dv num-classes)
               :children [param {:shape [num-feats] :value v}]})
            vecs))))

(defn serial-version [nodes]
  (let [op (->MultTensorOp)]
    (doseq [n nodes]
      (compute/forward-node-pass! op n))
    nodes))

(defn batch-version [nodes]
  (let [op (->MultTensorOp)
        sig (compute/batch-signature op (first nodes))]
    (compute/batch-forward-node-pass! op sig nodes)))

