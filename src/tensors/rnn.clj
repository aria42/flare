(ns tensors.rnn
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.core :as tensors]
            [tensors.graph-ops :as go]
            [tensors.computation-graph :as cg]))

(defprotocol RNNCell
  (add-input! [this input last-hidden]))

(s/defn lstm-cell
  [m :- model/PModel input-dim :- s/Int hidden-dim :- s/Int]
  (cg/with-scope "lstm"
   (let [W-shape [hidden-dim (+ hidden-dim input-dim)]
         forget-W (model/add-params! m W-shape :name "forget-W")
         forget-b (model/add-params! m [hidden-dim] :name "forget-b")
         keep-W (model/add-params! m W-shape :name "keep-W")
         keep-b (model/add-params! m [hidden-dim] :name "keep-b")
         hidden-W (model/add-params! m W-shape :name "hidden-W")
         hidden-b (model/add-params! m [hidden-dim] :name "hidden-b")
         out-W (model/add-params! m W-shape :name "out-W")
         out-b (model/add-params! m [hidden-dim] :name "out-b")
         last-cell-state* (atom nil)]
     (reify RNNCell
       (add-input! [this input last-hidden]
         (when-not (= (:shape input) [input-dim])
           (throw (ex-info "Input doesn't match expected shape"
                           {:expected-shape [input-dim]
                            :actual-shape (:shape input)})))
         (when-not (= (:shape last-hidden) [hidden-dim])
           (throw (ex-info "Hidden state doesn't match expected shape"
                           {:expected-shape [hidden-dim]
                            :actual-shape (:shape last-hidden)})))
         (let [x (go/concat 0 last-hidden input)
               forgot-probs (go/sigmoid (go/+ (go/* forget-W x) forget-b))
               keep-probs (go/sigmoid (go/+ (go/* keep-W x) keep-b))
               cell-state (go/tanh (go/+ (go/* hidden-W x) hidden-b))
               cell-state (go/+ (go/hadamard forgot-probs @last-cell-state*)
                            (go/hadamard keep-probs cell-state))
               out-probs (go/sigmoid (go/+ (go/* out-W x) out-b))]
           (reset! last-cell-state* cell-state)
           (go/hadamard out-probs (go/tanh cell-state))))))))
