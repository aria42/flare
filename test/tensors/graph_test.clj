(ns tensors.graph-test
  (:require [tensors.graph :refer :all]
            [clojure.test :refer :all]))

(deftest post-order-nodes-test
  (let [g {:node :a :children [{:node :b} {:node :c}]}]
    (is [:b :c :a] (map :node (post-order-nodes g)))))
