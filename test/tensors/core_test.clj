(ns tensors.core-test
  (:require [clojure.test :refer :all]
            [tensors.core :refer :all]))

(deftest guess-shape-test
  (testing "simple guess-shape"
    (are [x y] (= (guess-shape x) y)
      [1 2 3] [3]
      [[1 2 3]] [1 3]
      [[1 2] [3 4]] [2 2]))
  (testing "bad shapes should throw"
    (let [bad-shapes [[[1 2] [3]]]]
      (doseq [s bad-shapes]
        (is (thrown? RuntimeException (guess-shape s)))))))


(deftest effective-dimension-test
  (are [x y] (= (effective-dimension x) y)
    [1] 0
    [1 3] 1
    [2 3] 2
    [2 3 4 5] 4
    [1 2 1] 1))

(deftest vector-shape-test
  (are [x y] (= (vector-shape? x) y)
    [1] true
    [1 3] false
    [10] true))

(deftest scaler-shape-test
  (are [x y] (= (scalar-shape? x) y)
    [1] true
    [1 3] false
    [10] false))
