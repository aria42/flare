(ns tensors.module-test
  (:refer-clojure :exclude [comp])
  (:require [tensors.module :refer :all]
            [clojure.test :refer :all]
            [tensors.model :as model]
            [tensors.neanderthal-ops :as no]
            [tensors.node :as node]
            [tensors.compute :as compute]
            [tensors.computation-graph :as cg])
  (:import [org.apache.commons.math3.util FastMath]))

(deftest affine-test
  (let [factory (no/factory)
        m (model/simple-param-collection factory)
        aff (affine m 3 [2])
        x (node/constant factory "x" [1 2])]
    (model/fix-param! m "affine/W" [[1 2] [3 4] [5 6]])
    (model/fix-param! m "affine/b" [1 1 1])
    (is (= [6.0 12.0 18.0]
           (-> (compute/forward-pass! factory (graph aff x))
               :value
               seq)))))

(deftest from-op-test
  (let [s (from-op (cg/scalar-op :sigmoid))
        factory (no/factory)
        x (node/constant factory "x" [1 2])]
    (is (=
         [(/ 1.0 (+ 1.0 (Math/exp -1.0)))
          (/ 1.0 (+ 1.0 (Math/exp -2.0))) ]
         (seq
          (:value
           (compute/forward-pass! (no/factory) (graph s x))))))))

(deftest comp-test
  (let [s (from-op (cg/scalar-op :sigmoid))
        t (from-op (cg/scalar-op :tanh))
        m (comp t s)
        factory (no/factory)
        x (node/constant factory "x" [1 2])]
    (is (= [(FastMath/tanh (/ 1.0 (+ 1.0 (FastMath/exp -1.0))))
          (FastMath/tanh (/ 1.0 (+ 1.0 (FastMath/exp -2.0)))) ]
         (seq
          (:value
           (compute/forward-pass! factory (graph m x))))))
    (:value (compute/forward-pass!  factory (graph m x)))))
