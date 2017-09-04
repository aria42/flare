(ns tensors.computation-graph-test
  (:refer-clojure :exclude [+ *])
  (:require [clojure.test :refer :all]
            [tensors.computation-graph :refer :all]
            [tensors.core :as tensors]))

(deftest scope-test
  (testing "nested scope"
    (with-scope :affine
      (with-scope :logistic
        (let [Y (input "Y" [10 1])]
          (are [x y] (= x y)
            "affine/logistic/Y" (:ref-name Y)))))))
