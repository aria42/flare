(ns tensors.params-test
  (:require [tensors.params :refer :all]
            [clojure.test :refer :all]
            [tensors.neanderthal-ops :as no]
            [tensors.core :as tensors]
            [tensors.computation-graph :as cg]))

(deftest uniform-test
  (let [oracle (java.util.Random. 0)
        spec {:distribution :uniform :lower 0.0 :upper 1.0 :rand-seed 0}
        get-param (cg/get-param-rng spec)]
    (are [x y] (= x y)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param))))

(deftest init-params-test
  (let [spec {:distribution :uniform :lower -1.0 :upper 1.0 :rand-seed 0}
        get-param (cg/get-param-rng spec)
        init (init-params [2 2] get-param)]
    (are [x] (= (tensors/guess-shape (init-params x get-param)) x)
      [2 2]
      [10]
      [3 2 3]
      [3 1])))
