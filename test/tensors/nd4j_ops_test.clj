(ns tensors.nd4j-ops-test
  (:require [tensors.nd4j-ops :refer :all]
            [clojure.test :refer :all]
            [tensors.compute :as compute])
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray]))

(deftest sum-tensor-op-test
  (testing "C = A + B"
    (let [op (->SumTensorOp)
          A {:shape [2 2]
             :value (Nd4j/create (double-array [1 2 3 4]) (int-array [2 2]))
             :grad (Nd4j/zeros 2 2)}
          B {:shape [2 2]
             :value (Nd4j/create (double-array [1 2 3 4]) (int-array [2 2]))
             :grad (Nd4j/zeros 2 2)}
          C {:shape [2 2]
             :value (Nd4j/zeros 2 2)
             :grad (Nd4j/zeros 2 2)}
          node (assoc C :children [A, B])]
      (compute/ensure-valid?! op [A B])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value C) (Nd4j/create (double-array [2 4 6 8]) (int-array [2 2])))))
      ;; populate gradient of output
      (.assign ^INDArray (:grad C) (Nd4j/ones  2 2))
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad B) (Nd4j/ones 2 2)))
        (is (= (:grad A) (Nd4j/ones 2 2)))))))

(deftest mult-tensor-op
  (testing "C = A B"
    (let [op (->MultTensorOp)
          A {:shape [2 1]
             :value (Nd4j/create (double-array [1 2]) (int-array [2 1]))
             :grad (Nd4j/zeros 2 1)}
          B {:shape [1 2]
             :value (Nd4j/create (double-array [3 4]) (int-array [1 2]))
             :grad (Nd4j/zeros 1 2)}
          C {:shape [2 2]
             :value (Nd4j/zeros 2 2)
             :grad (Nd4j/zeros 2 2)}
          node (assoc C :children [A, B])]
      (compute/ensure-valid?! op [A B])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value C) (Nd4j/create (double-array [3 6 4 8]) (int-array [2 2]) \f))))
      ;; populate gradient of output
      (.assign ^INDArray (:grad C) (Nd4j/ones 2 2))
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad B) (Nd4j/create (double-array [3 3]) (int-array [1 2]))))
        (is (= (:grad A) (Nd4j/create (double-array [7 7]) (int-array [2 1]))))))))

(deftest squeeze-op
  (testing "B = (squeeze A 1)"
    (let [op (->SqueezeTensorOp)
          A {:shape [2 1]
             :value (Nd4j/ones 2 1)
             :grad (Nd4j/zeros 2 1)}
          B {:shape [2]
             :value (Nd4j/zeros 2)
             :grad (Nd4j/ones 2)}
          node (assoc B :children [A])]
      (compute/ensure-valid?! op [A])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value B) (Nd4j/ones 2))))
      ;; populate gradient of output
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad A) (Nd4j/ones 2 1)))))))

(deftest strech-op
  (testing "B = (strech A 1)"
    (let [op (->StrechTensorOp)
          A {:shape [2]
             :value (Nd4j/ones 2)
             :grad (Nd4j/zeros 2)}
          B {:shape [2 1]
             :value (Nd4j/zeros 2 1)
             :grad (Nd4j/ones 2 1)}
          node (assoc B :children [A])]
      (compute/ensure-valid?! op [A])
      (let [forward-node (compute/forward-node-pass! op node)]
        (is (= forward-node node))
        (is (= (:value B) (Nd4j/ones 2 1))))
      ;; populate gradient of output
      (let [backward-node (compute/backward-node-pass! op node)]
        (is (= backward-node node))
        (is (= (:grad A) (Nd4j/ones 2)))))))

(defn l2-dist [a b]
  (->> (map (fn [x y] (Math/pow (double (- x y)) 2.0)) a b)
       (reduce +)
       Math/sqrt))

(deftest cross-entropy-loss-op
  (testing "loss = (cross-entropy scores label)"
    (let [op (->CrossEntropyLossTensorOp)
          scores {:shape [3]
                  :value (Nd4j/create (double-array [1 1.5 2]) (int-array [3]))
                  :grad (Nd4j/zeros 3)}
          label {:shape [1]
                 :value (Nd4j/ones 1)}
          loss {:shape [1]
                :value (Nd4j/zeros 1)
                :grad (Nd4j/ones 1)}
          node (assoc loss :children [scores label])]
      (compute/ensure-valid?! op [scores label])
      (let [node (compute/prep op node)
            forward-node (compute/forward-node-pass! op node)]
        (is (<
             (l2-dist
              (:tensors.neanderthal-ops/probs forward-node)
              (Nd4j/create (double-array [ 0.186 0.307 0.506]) (int-array [3])))
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
