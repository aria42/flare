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
        zero  (flare/zeros (no/factory) [10])
        init-output (node/const "h0" zero)
        init-state (node/const "c0"  zero)
        input (node/const "x"  (repeat 50 1))
        [output state] (flare.rnn/add-input! cell input init-output init-state)]
    (is (= (:shape output) [10]))
    (is (= (:shape state) [10]))
    (is (= [10] (flare/shape (:value (compute/forward-pass! output)))))))
