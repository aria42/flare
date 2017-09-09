(ns tensors.model-test
  (:require [tensors.model :refer :all]
            [clojure.test :refer :all]
            [tensors.neanderthal-ops :as no]
            [tensors.core :as tensors]
            [tensors.computation-graph :as cg]
            [tensors.neanderthal-ops :as no]))

(deftest uniform-test
  (let [oracle (java.util.Random. 0)
        spec {:distribution :uniform :lower 0.0 :upper 1.0 :rand-seed 0}
        get-param (get-param-rng spec)]
    (are [x y] (= x y)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param)
      (.nextDouble oracle) (get-param))))

(deftest init-params-test
  (let [spec {:distribution :uniform :lower -1.0 :upper 1.0 :rand-seed 0}
        get-param (get-param-rng spec)
        init (init-params [2 2] get-param)]
    (are [x] (= (tensors/guess-shape (init-params x get-param)) x)
      [2 2]
      [10]
      [3 2 3]
      [3 1])))


(deftest param-collection-test
  (testing "simple param collection test"
    (let [m (simple-param-collection (no/->Factory))
          n (add-params! m  [2 2] :name "W")]
      (is (= n (canonical-node m "W")))
      (is (= n (get (into {} (seq m)) "W")))))
  (testing "disallow repeated names"
    (let [m (simple-param-collection (no/->Factory))]
      (add-params! m [2 2] :name "W")
      (is (thrown? Exception (add-params! m [2 2] "W")))))
  (testing "can init!"
    (let [m (simple-param-collection (no/->Factory))]
      (add-params! m [2 2] :name "W1")
      (add-params! m [2 2] :name "W2")
      (doseq [k ["W1" "W2"]]
        (is (:value (canonical-node m k)))
        (is (:grad (canonical-node m k)))))))
