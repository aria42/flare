(ns flare.rnn-test
  (:require [flare.rnn :as rnn :refer :all]
            [clojure.test :as t :refer :all]
            [flare.model :as model]
            [flare.neanderthal-ops :as no]
            [flare.computation-graph :as cg]
            [flare.node :as node]
            [flare.core :as flare]
            [flare.compute :as compute]))

(flare/set! {:eager? false :factory (no/factory)})

(deftest lstm-cell-test
  (let [m (model/simple-param-collection)
        cell (flare.rnn/lstm-cell m 50 10)
        zero  (flare/zeros [10])
        h (init-hidden cell)
        input (node/const "x"  (repeat 50 1))
        [output state] (flare.rnn/add-input cell input h)]
    (is (= (:shape output) [10]))
    (is (= (:shape state) [10]))
    (is (= [10] (flare/shape (:value (compute/forward-pass! output)))))))

(deftest lstm-seq
  (let [m (model/simple-param-collection)
        cell (flare.rnn/lstm-cell m 50 10)
        zero (flare/zeros [10])
        h (init-hidden cell)
        input (node/const "x" (repeat 50 1))
        inputs (repeat 500 input)
        hiddens (flare.rnn/build-seq cell inputs)
        last-output (first (last hiddens))
        last-hidden (second (last hiddens))]
    (is (= (count hiddens) 500))
    (is (every? #(= (count %) 2) hiddens))
    (is (= [10] (flare/shape (:value (compute/forward-pass! last-output)))))))
