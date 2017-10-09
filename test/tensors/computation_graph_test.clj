(ns tensors.computation-graph-test
  (:refer-clojure :exclude [+ * concat])
  (:require [clojure.test :refer :all]
            [tensors.computation-graph :refer :all]
            [tensors.core :as tensors]
            [tensors.node :as node]))

(deftest scope-test
  (testing "nested scope"
    (node/with-scope :affine
      (node/with-scope :logistic
        (let [Y (node/input "Y" [10 1])]
          (are [x y] (= x y)
            "affine/logistic/Y" (:ref-name Y)))))))

(deftest arithcmetic-test
  (testing "simple addition"
    (let [Y (node/input "Y" [1 10])
          X (node/input "X" [1 10])
          L (node/input "L" [1 5])
          Z (+ X Y)]
      (are [k v] (= (get Z k) v)
        :shape [1 10]
        :graph-op (->SumGraphOp)
        :children [X Y])
      (is (thrown? RuntimeException (+ X L)))))
  (testing "simple multiplication"
    (let [Y (node/input "Y" [1 10])
          X (node/input "X" [10 1])
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

(deftest hadamard-test
  (testing "hadamard"
    (let [X (node/input "X" [5 5])
          Y (node/input "Y" [5 5])]
      (is (= [5 5] (:shape (hadamard X Y))))
      (is (thrown? RuntimeException (hadamard X (node/input [5 1])))))))


(deftest logistic-regression-test
  (testing "create logistic regression graph (make parameters inputs)"
    (let [num-classes 2
          num-feats 10
          W (node/constant "W" [num-classes num-feats] nil)
          b (node/constant "bias" [num-classes] nil)
          feat-vec (node/input "f" [num-feats])
          activations (+ (* W feat-vec) b)
          label (node/input "label" [1])
          loss (cross-entropy-loss activations label)]
      (is (tensors/scalar-shape? (:shape loss))))))

(deftest concat-op-test
  (testing "concat op"
    (let [op (->ConcatOp 0)
          inputs [(node/input [2 4]) (node/input [3 4])]]
      (is (= [5 4] (forward-shape op inputs)))
      (op-validate! op inputs)
      (is (thrown? RuntimeException
                   (op-validate! op [(node/input [3 4]) (node/input [3 3])]))))))

