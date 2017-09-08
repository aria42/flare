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

(deftest mult-tensor-op
  (testing "C = A B"
    (let [op (->MultTensorOp)
          A {:shape [2 1]
             :value (dge 2 1 [1 2])
             :grad (dge 2 1)}
          B {:shape [1 2]
             :value (dge 1 2 [3 4])
             :grad (dge 1 2)}
          C {:shape [2 2]
             :value (dge 2 2)
             :grad (dge 2 2)}
          node (assoc C :children [A, B])]
      (compute/ensure-valid?! op [A B])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value C) (dge 2 2 [3 6 4 8]))))
      ;; populate gradient of output
      (copy! (dge 2 2 [1 1 1 1]) (:grad C))
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad B) (dge 1 2 [3 3])))
        (is (= (:grad A) (dge 2 1 [7 7])))))))

(deftest squeeze-op
  (testing "B = (squeeze A 1)"
    (let [op (->SqueezeTensorOp)
          A {:shape [2 1]
             :value (dge 2 1 [1 1])
             :grad (dge 2 1 [1 1])}
          B {:shape [2]
             :value (dv 2)
             :grad (dv [1 1])}
          node (assoc B :children [A])]
      (compute/ensure-valid?! op [A])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value B) (dv [1 1]))))
      ;; populate gradient of output
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad A) (dge 2 1[1 1])))))))


(deftest strech-op
  (testing "B = (strech A 1)"
    (let [op (->StrechTensorOp)
          A {:shape [2]
             :value (dv [1 1])
             :grad (dv [0 0])}
          B {:shape [2 1]
             :value (dge 2 1 [1 1])
             :grad (dge 2 1 [1 1])}
          node (assoc B :children [A])]
      (compute/ensure-valid?! op [A])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value B) (dge 2 1 [1 1]))))
      ;; populate gradient of output
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad A) (dv [1 1])))))))
