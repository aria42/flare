(ns tensors.module-test
  (:refer-clojure :exclude [comp])
  (:require [tensors.module :refer :all]
            [clojure.test :refer :all]
            [tensors.model :as model]
            [tensors.neanderthal-ops :as no]
            [tensors.node :as node]
            [tensors.compute :as compute]
            [tensors.computation-graph :as cg]))

(deftest affine-test
  (let [factory (no/->Factory)
        m (model/simple-param-collection factory)
        aff (affine m 3 [2])
        x (node/constant "x" factory [1 2])]
    (model/fix-param! m "affine/W" [[1 2] [3 4] [5 6]])
    (model/fix-param! m "affine/b" [1 1 1])
    (is (= [6.0 12.0 18.0]
           (-> (graph aff x) (compute/forward-pass! m) :value seq)))))

(deftest from-op-test
  (let [s (from-op (cg/scalar-op :sigmoid))
        factory (no/->Factory)
        x (node/constant factory [1 2])]
    (is (=
         [(/ 1.0 (+ 1.0 (Math/exp -1.0)))
          (/ 1.0 (+ 1.0 (Math/exp -2.0))) ]
         (seq
          (:value
           (compute/forward-pass!
            (graph s [x])
            (model/simple-param-collection (no/->Factory)))))))))

(deftest comp-test
  (let [s (from-op (cg/scalar-op :sigmoid))
        t (from-op (cg/scalar-op :tanh))
        m (comp t s)
        factory (no/->Factory)
        model (model/simple-param-collection factory)
        x (node/constant factory [1 2])]
    (is (=
         [(Math/tanh (/ 1.0 (+ 1.0 (Math/exp -1.0))))
          (Math/tanh (/ 1.0 (+ 1.0 (Math/exp -2.0)))) ]
         (seq
          (:value
           (compute/forward-pass! (graph m x) model)))))
    (:value (compute/forward-pass! (graph m x) model))))

