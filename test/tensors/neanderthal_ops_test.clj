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
        (is (= (:grad A) (dge 2 1 [7 7]))))))
  (testing "c = A b (matrix * vector)"
    (let [op (->MultTensorOp)
          A {:shape [2 3]
             :value (dge 2 3 [1 2 3 4 5 6])
             :grad (dge 2 3)}
          b {:shape [3]
             :value (dv [1 1 1])
             :grad (dv 3)}
          c {:shape [2]
             :value (dv 2)
             :grad (dv 2)}
          node (assoc c :children [A, b])]
      (compute/ensure-valid?! op [A b])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value c) (dv [9 12]))))
      ;; populate gradient of output
      (copy! (dv [1 1]) (:grad c))
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad b) (dv [3 7 11])))
        (is (= (:grad A) (dge 2 3 [1 1 1 1 1 1])))))))

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

(defn l2-dist [a b]
  (->> (map (fn [x y] (Math/pow (double (- x y)) 2.0)) a b)
       (reduce +)
       Math/sqrt))

(deftest cross-entropy-loss-op
  (testing "loss = (cross-entropy scores label)"
    (let [op (->CrossEntropyLossTensorOp)
          scores {:shape [3]
                  :value (dv [1 1.5 2])
                  :grad (dv [0 0 0])}
          label {:shape [1]
                 :value (dv [1])}
          loss {:shape [1]
                :value (dv [0.0])
                :grad (dv [1.0])}
          node (assoc loss :children [scores label])]
      (compute/ensure-valid?! op [scores label])
      (let [node (compute/prep op node)
            forward-node (compute/forward-node-pass! op node)]
        (is (<
             (l2-dist
              (:tensors.neanderthal-ops/probs forward-node)
              (dv [ 0.186 0.307 0.506]))
             0.001))
        (let [loss-scalar (first (:value forward-node))
              target-loss (- (Math/log 0.307))]
          (is (< (Math/abs (- target-loss loss-scalar)) 0.001)))
        ;; populate gradient of output
        (let [backward-node (compute/backward-node-pass! op forward-node)]
          (is (<
               (l2-dist
                (:grad scores)
                [0.186 -0.693 0.506])
               0.001)))))))
