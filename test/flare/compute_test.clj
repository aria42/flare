(ns flare.compute-test
  (:require [flare.compute :refer :all]
            [flare.computation-graph :as cg]
            [flare.core :as flare]
            [flare.neanderthal-ops :as no]
            [flare.node :as node]
            [flare.model :as model]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.test :refer :all]
            [flare.compute :as compute]))

(deftest compile-forward-test
  (testing "simple graph"
    (let [X (node/input "X" [2 2])
          Y (node/input "Y" [2 2])
          Z (cg/+ X Y)
          factory (no/factory)
          input-vals {"X" [[1 2] [2 1]] "Y" [[1 2] [1 1]]}]
      (with-inputs! factory Z input-vals)
      (let [Z (forward-pass! factory Z)]
        (is (= [[2.0 4.0] [3.0 2.0]]
               (seq (:value Z))))
        (is (= [[0.0 0.0] [0.0 0.0]]
               (seq (:grad Z)))))))
  (testing "lr graph"
    (let [num-classes 2
          num-feats 3
          factory (no/factory)
          m (model/simple-param-collection factory)
          W (model/add-params! m [num-classes num-feats] :name "W")
          b (model/add-params! m [num-classes] :name "b")
          feat-vec (cg/strech (node/input "f" [num-feats]) 1)
          activations (cg/squeeze (cg/+ (cg/* W feat-vec) (cg/strech b 1)) 1)
          ;; keep 1 as the "correct" label
          label (node/input "label" [1])
          loss (cg/cross-entropy-loss activations label)]
      (let [input->vals {"f" [1 2 1] "label" [0]}
            one-grad (flare/from factory [1.0])
            _ (compute/with-inputs! factory loss input->vals)
            loss (forward-pass! factory loss)]
        (is (not (neg? (first (seq (:value loss))))))
        (backward-pass! factory (assoc loss :grad one-grad))
        (let [W-grad (:grad (model/canonical-node m "W"))
              [wrong-row right-row] (rows W-grad)]
          ;; the incorrect label should get neg gradient
          (is (every? neg? wrong-row))
          ;; the correct label should get neg gradient
          (is (every? pos? right-row)))
        loss)))
  (testing "repeated test"
    (let [factory (no/factory)
          model (model/simple-param-collection factory)
          params-map (-> model meta :data)
          ;; need to override value 
          X (model/add-params! model [2])
          Z (cg/+ X X)]
      ;; hack to set values for params
      (flare/copy! factory
         (:value (get params-map (:ref-name X)))
         [2.0 2.0])
      (let [Z (forward-pass!  factory Z)]
        (is (= [4.0 4.0] (seq (:value Z))))
        (backward-pass! factory (assoc Z :grad (flare/from factory [1 1])))
        (is (= [2.0 2.0] (seq (:grad X))))))))

