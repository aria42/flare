(ns tensors.compute-test
  (:require [tensors.compute :refer :all]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.graph-ops :as go]
            [tensors.neanderthal-ops :as no]
            [tensors.model :as model]
            [clojure.test :refer :all]))

(deftest compile-forward-test
  (testing "simple graph"
    (let [X (cg/input "X" [2 2])
          Y (cg/input "Y" [2 2])
          Z (go/+ X Y)
          model (model/simple-param-collection)
          factory (no/->Factory)
          Z (compile-graph Z factory model)
          input-vals {"X" [[1 2] [2 1]] "Y" [[1 2] [1 1]]}]
      (forward-pass! Z factory input-vals)
      (is (= [[2.0 4.0] [3.0 2.0]]
             (tensors/->clj factory (:value Z)))))))

