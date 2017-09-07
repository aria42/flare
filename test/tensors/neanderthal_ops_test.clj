(ns tensors.neanderthal-ops-test
  (:require [tensors.neanderthal-ops :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.test :refer :all]
            [tensors.compute :as compute]))

(deftest sum-tensor-op-test
  (testing "C = A + B"
    (let [op (->SumTensorOp)
          A {:shape [2 2]
             :value (dge 2 2[1 2 3 4])
             :grad (dge 2 2)}
          B {:shape [2 2]
             :value (dge 2 2[1 2 3 4])
             :grad (dge 2 2)}
          C {:shape [2 2]
             :value (dge 2 2)
             :grad (dge 2 2)}
          node (assoc C :children [A, B])]
      (compute/ensure-valid?! op [A B])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value C) (dge 2 2 [2 4 6 8]))))
      ;; populate gradient of output
      (copy! (dge 2 2 [1 1 1 1]) (:grad C))
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad B) (dge 2 2 [1 1 1 1])))
        (is (= (:grad A) (dge 2 2 [1 1 1 1])))))))
