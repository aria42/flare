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
  (flare/set! {:factory (no/factory)})
  (testing "simple graph"
    (let [X (node/constant "X" [[1 2] [2 1]])
          Y (node/constant "Y" [[1 2] [1 1]])
          Z (cg/+ X Y)]
      (let [Z (forward-pass! Z)]
        (is (= [[2.0 4.0] [3.0 2.0]]
               (seq (:value Z))))
        (is (= [[0.0 0.0] [0.0 0.0]]
               (seq (:grad Z)))))))
  (testing "lr graph"
    (let [num-classes 2
          num-feats 3
          m (model/simple-param-collection)
          W (model/add-params! m [num-classes num-feats] :name "W")
          b (model/add-params! m [num-classes] :name "b")
          feat-vec (node/constant "f" [1 2 1])
          activations (cg/+ (cg/* W feat-vec) b)
          ;; keep 1 as the "correct" label
          label (node/constant "label" [0])
          loss (cg/cross-entropy-loss activations label)
          factory (:factory (flare/state))]
      (let [one-grad (flare/from factory [1.0])
            loss (forward-pass! loss)]
        (is (not (neg? (double (first (seq (:value loss)))))))
        (backward-pass! (assoc loss :grad one-grad))
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

