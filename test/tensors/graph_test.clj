(ns tesnors.graph-test
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer :all]
            [tensors.graph :refer :all]))

(deftest scope-test
  (testing "nested scope"
    (with-scope :affine
      (with-scope :logistic
        (let [Y (input "Y" [10 1])]
          (are [x y] (= x y)
            "affine/logistic/Y" (:ref-name Y)))))))

(deftest arithcmetic-test
  (testing "simple addition"
    (let [Y (input "Y" [1 10])
          X (input "X" [1 10])
          L (input "L" [1 5])
          Z (+ X Y)]
      (are [k v] (= (get Z k) v)
        :shape [1 10]
        :graph-op (->SumGraphOp)
        :children [X Y]
        :input-names #{"X" "Y"})
      (is (thrown? RuntimeException (+ X L)))))
  (testing "simple multiplication"
    (let [Y (input "Y" [1 10])
          X (input "X" [10 1])
          Z (* X Y)
          Z-rev (* Y X)]
      (is (thrown? RuntimeException (* Z Y)))
      (are [k v] (= (get Z k) v)
        :shape [10 10]
        :graph-op (->MultGraphOp)
        :children [X Y]
        :input-names #{"X" "Y"})
      (are [k v] (= (get Z-rev k) v)
        :shape [1 1]
        :graph-op (->MultGraphOp)
        :children [Y X]
        :input-names #{"X" "Y"}))))


(deftest logistic-regression-test
  (testing "create logistic regression graph"
    (let [num-classes 2
          num-feats 10 
          W (params "W" [num-classes num-feats])
          b (params "bias" [num-classes 1])
          feat-vec (input "f" [num-feats 1])
          activations (+ (* W feat-vec) b)
          probs (soft-max activations)
          label (input "label" [1])
          loss (cross-entropy-loss probs label)]
      (= (:shape loss) [1]))))
