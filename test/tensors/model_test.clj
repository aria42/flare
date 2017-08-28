(ns tensors.model-test
  (:require [tensors.model :refer :all]
            [clojure.test :refer :all]
            [tensors.neanderthal-ops :as no]
            [tensors.core :as tensors]))

(deftest uniform-test
  (let [oracle (java.util.Random. 0)
        spec {:type :uniform :lower 0.0 :upper 1.0 :rand-seed 0}
        get-param (get-param-rng spec)]
    (are [x y] (= x y)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param))))

(deftest init-params-test
  (let [spec {:type :uniform :lower -1.0 :upper 1.0 :rand-seed 0}
        get-param (get-param-rng spec)
        init (init-params [2 2] get-param)]
    (are [x] (= (tensors/guess-shape (init-params x get-param)) x)
      [2 2]
      [10]
      [3 2 3]
      [3 1])))

(deftest param-collection-test
  (let [model (simple-param-collection (no/->Factory))]
    (is (= (add-params! model "params" [2 2]
                     {:type :uniform :lower -1.0 :upper 1.0 :rand-seed 0})
        (get-params model "params")))))
