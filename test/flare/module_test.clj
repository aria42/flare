(ns flare.module-test
  (:refer-clojure :exclude [comp])
  (:require [flare.module :refer :all]
            [clojure.test :refer :all]
            [flare.model :as model]
            [flare.neanderthal-ops :as no]
            [flare.node :as node]
            [flare.compute :as compute]
            [flare.computation-graph :as cg]
            [flare.core :as flare])
  (:import [org.apache.commons.math3.util FastMath]))

(flare/set! {:eager? false :factory (no/factory)})

(deftest affine-test
  (let [factory (no/factory)
        m (model/simple-param-collection factory)
        aff (affine m 3 [2])
        x (node/const "x" [1 2])]
    (model/fix-param! m "W" [[1 2] [3 4] [5 6]])
    (model/fix-param! m "b" [1 1 1])
    (is (= [6.0 12.0 18.0]
           (-> (compute/forward-pass! factory (graph aff x))
               :value
               seq)))))

(deftest from-op-test
  (let [s (from-op (cg/scalar-op :sigmoid))
        factory (no/factory)
        x (node/const "x" [1 2])]
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
        x (node/const "x" [1 2])]
    (is (= [(FastMath/tanh (/ 1.0 (+ 1.0 (FastMath/exp -1.0))))
          (FastMath/tanh (/ 1.0 (+ 1.0 (FastMath/exp -2.0)))) ]
         (seq
          (:value
           (compute/forward-pass! factory (graph m x))))))
    (:value (compute/forward-pass!  factory (graph m x)))))
