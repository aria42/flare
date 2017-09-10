(ns tensors.compute-test
  (:require [tensors.compute :refer :all]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.graph-ops :as go]
            [tensors.neanderthal-ops :as no]
            [tensors.model :as model]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.test :refer :all]))

(deftest compile-forward-test
  (testing "simple graph"
    (let [X (cg/input "X" [2 2])
          Y (cg/input "Y" [2 2])
          Z (go/+ X Y)
          factory (no/->Factory)
          model (model/simple-param-collection factory)
          Z (compile-graph Z factory model)
          input-vals {"X" [[1 2] [2 1]] "Y" [[1 2] [1 1]]}]
      (forward-pass! Z input-vals)
      (is (= [[2.0 4.0] [3.0 2.0]]
             (tensors/->clj (:factory Z) (:value Z))))
      (is (= [[0.0 0.0] [0.0 0.0]]
             (tensors/->clj (:factory Z) (:grad Z))))))
  (testing "lr graph"
    (let [num-classes 2
          num-feats 3
          factory (no/->Factory)
          m (model/simple-param-collection factory)
          W (model/add-params! m [num-classes num-feats] :name "W")
          b (model/add-params! m [num-classes] :name "b")
          feat-vec (go/strech (cg/input "f" [num-feats]) 1)
          activations (go/squeeze (go/+ (go/* W feat-vec) (go/strech b 1)) 1)
          ;; keep 1 as the "correct" label
          label (cg/input "label" [1])
          loss (go/cross-entropy-loss activations label)
          loss (compile-graph loss factory m)]
      (let [input->vals {"f" [1 2 1] "label" [0]}
            one-grad (tensors/from-nums factory [1.0])
            loss (-> loss
                     (forward-pass! input->vals)
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
  )

