(ns tensors.graph-ops-test
  (:refer-clojure :exclude [+ *])
  (:require [tensors.graph-ops :refer :all]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [clojure.test :refer :all]))

(deftest arithcmetic-test
  (testing "simple addition"
    (let [Y (cg/input "Y" [1 10])
          X (cg/input "X" [1 10])
          L (cg/input "L" [1 5])
          Z (+ X Y)]
      (are [k v] (= (get Z k) v)
        :shape [1 10]
        :graph-op (->SumGraphOp)
        :children [X Y])
      (is (thrown? RuntimeException (+ X L)))))
  (testing "simple multiplication"
    (let [Y (cg/input "Y" [1 10])
          X (cg/input "X" [10 1])
          Z (* X Y)
          Z-rev (* Y X)]
      (is (thrown? RuntimeException (* Z Y)))
      (are [k v] (= (get Z k) v)
        :shape [10 10]
        :graph-op (->MultGraphOp)
        :children [X Y])
      (are [k v] (= (get Z-rev k) v)
        :shape [1 1]
        :graph-op (->MultGraphOp)
        :children [Y X]))))


(deftest logistic-regression-test
  (testing "create logistic regression graph"
    (let [num-classes 2
          num-feats 10
          W (cg/params "W" [num-classes num-feats])
          b (cg/params "bias" [num-classes 1])
          feat-vec (cg/input "f" [num-feats 1])
          activations (+ (* W feat-vec) b)
          probs (soft-max (squeeze activations 1))
          label (cg/input "label" [1])
          loss (cross-entropy-loss probs label)]
      (tensors/scalar-shape? (:shape loss)))))
