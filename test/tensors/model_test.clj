(ns tensors.model-test
  (:require [tensors.model :refer :all]
            [clojure.test :refer :all]
            [tensors.neanderthal-ops :as no]
            [tensors.core :as tensors]
            [tensors.computation-graph :as cg]
            [tensors.neanderthal-ops :as no]
            [tensors.model :as model]))

(deftest uniform-test
  (let [oracle (java.util.Random. 0)
        spec {:distribution :uniform :lower 0.0 :upper 1.0 :rand-seed 0}
        get-param (get-param-rng spec)]
    (are [x y] (= x y)
      (.nextDouble oracle) (get-param nil 0.0)
      (.nextDouble oracle) (get-param nil 0.0)
      (.nextDouble oracle) (get-param nil 0.0))))

(deftest param-collection-test
  (testing "simple param collection test"
    (let [m (simple-param-collection (no/factory))
          n (add-params! m  [2 2] :name "W")]
      (is (= n (canonical-node m "W")))
      (is (= n (get (into {} (seq m)) "W")))))
  (testing "disallow repeated names"
    (let [m (simple-param-collection (no/factory))]
      (add-params! m [2 2] :name "W")
      (is (thrown? Exception (add-params! m [2 2] "W")))))
  (testing "can init!"
    (let [m (simple-param-collection (no/factory))]
      (add-params! m [2 2] :name "W1")
      (add-params! m [2 2] :name "W2")
      (doseq [k ["W1" "W2"]]
        (is (:value (canonical-node m k)))
        (is (:grad (canonical-node m k)))))))

(deftest round-trip-doubles
  (testing "round trip dobules"
    (let [f (no/factory)
          m (simple-param-collection f)
          x (add-params! m [2 3] :name "x")
          y (add-params! m [3 2] :name "y")
          z (add-params! m [3] :name "z")]
      (tensors/copy-from-input! f (:value x) (range 6))
      (tensors/copy-from-input! f (:value y) (range 6 12))
      (tensors/copy-from-input! f (:value z) (range 12 15))
      ;; should be [0.0-15.0)
      (let [xs (model/to-doubles m)]
        (is (= (map double (range 15)) (seq xs))))
      ;; scramble values
      (model/rand-params! m)
      (is (not= (map double (range 15)) (to-doubles m)))
      ;; write values
      (from-doubles! m (double-array (range 42 (+ 42 15))))
      (is (= (map double (range 42 (+ 42 15))) (seq (to-doubles m)))))))
