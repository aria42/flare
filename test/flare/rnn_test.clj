(ns flare.rnn-test
  (:require [flare.rnn :as rnn :refer :all]
            [clojure.test :as t :refer :all]
            [flare.model :as model]
            [flare.neanderthal-ops :as no]
            [flare.computation-graph :as cg]
            [flare.node :as node]
            [flare.core :as flare]
            [flare.compute :as compute]))

(deftest lstm-cell-test
  (let [factory (no/factory)
        m (model/simple-param-collection factory)
        cell (flare.rnn/lstm-cell m 50 10)
        zero  (flare/zeros factory [10])
        init-output (node/constant  factory "h0" zero)
        init-state (node/constant factory "c0"  zero)
        input (node/constant factory "x"  (repeat 50 1))
        [output state] (flare.rnn/add-input! cell input init-output init-state)]
    (is (= (:shape output) [10]))
    (is (= (:shape state) [10]))
    (is (=
         [10]
         (flare/shape factory (:value (compute/forward-pass! factory output)))))))
