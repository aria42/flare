(ns tensors.neanderthal-ops-test
  (:require [tensors.neanderthal-ops :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.test :refer :all]
            [tensors.compute :as compute]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.node :as node]))

(deftest sum-tensor-op-test
  (testing "C = A + B"
    (let [op (->SumTensorOp)
          A (node/map->Node
             {:shape [2 2]
              :value (dge 2 2[1 2 3 4])
              :grad (dge 2 2)})
          B (node/map->Node
             {:shape [2 2]
              :value (dge 2 2[1 2 3 4])
              :grad (dge 2 2)})
          C (node/map->Node
             {:shape [2 2]
              :value (dge 2 2)
              :grad (dge 2 2)})
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
          A (node/map->Node
             {:shape [2 1]
              :value (dge 2 1 [1 2])
              :grad (dge 2 1)})
          B (node/map->Node
             {:shape [1 2]
              :value (dge 1 2 [3 4])
              :grad (dge 1 2)})
          C (node/map->Node
             {:shape [2 2]
              :value (dge 2 2)
              :grad (dge 2 2)})
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
          A (node/map->Node {:shape [2 3]
              :value (dge 2 3 [1 2 3 4 5 6])
              :grad (dge 2 3)})
          b (node/map->Node
             {:shape [3]
              :value (dv [1 1 1])
              :grad (dv 3)})
          c (node/map->Node
             {:shape [2]
              :value (dv 2)
              :grad (dv 2)})
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
          A (node/map->Node
             {:shape [2 1]
              :value (dge 2 1 [1 1])
              :grad (dge 2 1 [1 1])})
          B (node/map->Node
             {:shape [2]
              :value (dv 2)
              :grad (dv [1 1])})
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
          A (node/map->Node
             {:shape [2]
              :value (dv [1 1])
              :grad (dv [0 0])})
          B (node/map->Node
             {:shape [2 1]
              :value (dge 2 1 [1 1])
              :grad (dge 2 1 [1 1])})
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

(deftest elemen-transform-test
  (let [x {:ref-name "x" :shape [3] :value (dv [1 2 3]) :grad (dv [2 2 2])}
        fx {:ref-name "fx" :shape [3] :value (dv 3) :grad (dv 3)}]
    (testing "exp(x) test"
      (let [op (tensors/get-op (->Factory) :exp)]
        (compute/forward-node-pass! op (assoc fx :children [x]))
        (is (:value fx) (dv (map #(Math/exp %) [1 2 3])))
        (compute/backward-node-pass! op (assoc fx :children [x]))
        (is (:grad x) (dv (map #(* 2 (Math/exp %)) [1 2 3])))))))

(deftest hadamard-test
  (testing "base hadamard test"
    (let [out (dv 3)]
      (hadamard out (dv [1 2 3]) (dv [1 2 2]))
      (is out (dv [1 4 6])))
    (let [out (dge 2 2)
          x (dge 2 2 [1 2 3 4])]
      (hadamard out x x)
      (is (= out (dge 2 2 [1 4 9 16])))))
  (testing "hadamard op test"
    (let [op (->HadamardTensorOp)
          x {:ref-name "x" :value (dv [1 2 3]) :grad (dv 3)}
          y {:ref-name "x" :value (dv [2 2 2]) :grad (dv 3)}
          n {:ref-name "n" :value (dv 3) :children [x y] :grad (dv [3 3 3])}]
      (compute/forward-node-pass! op n)
      (is (= (dv [2 4 6]) (:value n)))
      (compute/backward-node-pass! op n)
      (is (= (dv [6 6 6]) (:grad x)))
      (is (= (dv [3 6 9]) (:grad y))))))

(deftest cross-entropy-loss-op
  (testing "loss = (cross-entropy scores label)"
    (let [op (->CrossEntropyLossTensorOp)
          scores (node/map->Node
                  {:shape [3]
                   :value (dv [1 1.5 2])
                   :grad (dv [0 0 0])})
          label (node/map->Node
                 {:shape [1]
                  :value (dv [1])})
          loss (node/map->Node
                {:shape [1]
                 :value (dv [0.0])
                 :grad (dv [1.0])})
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

(deftest concat-test
  (testing "concat forward test"
    (let [n1 (node/map->Node {:shape [2] :value (dv [1 2])})
          n2 (node/map->Node {:shape [3] :value (dv [3 4 5])})
          o (node/map->Node {:shape [5] :value (dv 5) :children [n1 n2]})
          op (->ConcatTensorOp 0)]
      (compute/forward-node-pass! op o)
      (is (= (dv [1 2 3 4 5])(:value o))))
    (let [n1 (node/map->Node {:shape [1 2] :value (dge 1 2 [1 2])})
          n2 (node/map->Node {:shape [1 3] :value (dge 1 3 [3 4 5])})
          o (node/map->Node {:shape [1 5] :value (dge 1 5) :children [n1 n2]})
          op (->ConcatTensorOp 1)]
      (compute/forward-node-pass! op o)
      (is (= (dge 1 5 (range 1 6))(:value o)))))
  (testing "concat backward test"
    (let [n1 (node/map->Node {:shape [2] :grad (dv 2)})
          n2 (node/map->Node {:shape [3] :grad (dv 3)})
          o (node/map->Node {:shape [5] :grad (dv (range 1 6)) :children [n1 n2]})
          op (->ConcatTensorOp 0)]
      (compute/backward-node-pass! op o)
      (is (= (dv [1 2])(:grad n1)))
      (is (= (dv [3 4 5])(:grad n2))))
    (let [n1 (node/map->Node {:shape [1 2] :grad (dge 1 2)})
          n2 (node/map->Node {:shape [1 3] :grad (dge 1 3)})
          o (node/map->Node {:shape [1 5] :grad (dge 1 5 (range 1 6)) :children [n1 n2]})
          op (->ConcatTensorOp 1)]
      (compute/backward-node-pass! op o)
      (is (= (dge 1 2 [1 2]) (:grad n1)))
      (is (= (dge 1 3 [3 4 5]) (:grad n2))))))
