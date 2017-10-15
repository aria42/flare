(ns tensors.compute-test
  (:require [tensors.compute :refer :all]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.neanderthal-ops :as no]
            [tensors.node :as node]
            [tensors.model :as model]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.test :refer :all]))

(deftest compile-forward-test
  (testing "simple graph"
    (let [X (node/input "X" [2 2])
          Y (node/input "Y" [2 2])
          Z (cg/+ X Y)
          factory (no/->Factory)
          model (model/simple-param-collection factory)
          input-vals {"X" [[1 2] [2 1]] "Y" [[1 2] [1 1]]}]
      (let [Z (forward-pass! Z model input-vals)]
        (is (= [[2.0 4.0] [3.0 2.0]]
               (tensors/->clj factory (:value Z))))
        (is (= [[0.0 0.0] [0.0 0.0]]
               (tensors/->clj factory (:grad Z)))))))
  (testing "lr graph"
    (let [num-classes 2
          num-feats 3
          factory (no/->Factory)
          m (model/simple-param-collection factory)
          W (model/add-params! m [num-classes num-feats] :name "W")
          b (model/add-params! m [num-classes] :name "b")
          feat-vec (cg/strech (node/input "f" [num-feats]) 1)
          activations (cg/squeeze (cg/+ (cg/* W feat-vec) (cg/strech b 1)) 1)
          ;; keep 1 as the "correct" label
          label (node/input "label" [1])
          loss (cg/cross-entropy-loss activations label)]
      (let [input->vals {"f" [1 2 1] "label" [0]}
            one-grad (tensors/from-nums factory [1.0])
            loss (-> loss
                     (forward-pass! m input->vals)
                     (assoc :grad one-grad))]
        (is (not (neg? (first (tensors/->clj factory (:value loss))))))
        (backward-pass! loss)
        (let [W-grad (:grad (model/canonical-node m "W"))
              [wrong-row right-row] (rows W-grad)]
          ;; the incorrect label should get neg gradient
          (is (every? neg? wrong-row))
          ;; the correct label should get neg gradient
          (is (every? pos? right-row)))
        loss))
    )
  (testing "repeated test"
    (let [factory (no/->Factory)
          model (model/simple-param-collection factory)
          params-map (-> model meta :data)
          ;; need to override value 
          X (model/add-params! model [2])
          Z (cg/+ X X)]
      ;; hack to set values for params
      (tensors/copy-from-input! factory
         (:value (get params-map (:ref-name X)))
         [2.0 2.0])
      (let [Z (forward-pass! Z model)]
        (is (= [4.0 4.0] (tensors/->clj factory (:value Z))))
        (backward-pass! (assoc Z :grad (tensors/from-nums factory [1 1])))
        (is (= [2.0 2.0] (tensors/->clj factory (:grad X))))))))

