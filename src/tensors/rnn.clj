(ns tensors.rnn
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.core :as tensors]
            [tensors.node :as node]
            [tensors.computation-graph :as cg]
            [tensors.module :as module]))

(defprotocol RNNCell
  (cell-model [this])
  (output-dim [this])
  (add-input! [this input last-output last-state]))

(s/defn lstm-cell
  [m :- model/PModel input-dim :- s/Int hidden-dim :- s/Int]
  (node/with-scope "lstm"
    (let [sigmoid (module/from-op (cg/scalar-op :sigmoid))
          mk-affine (fn [scope]
                      (node/with-scope scope
                        (module/affine m hidden-dim [(+ input-dim hidden-dim)])))
          keep (module/comp sigmoid (mk-affine "keep"))
          forget (module/comp sigmoid (mk-affine "forget"))
          output (module/comp sigmoid (mk-affine "output"))
          tanh (module/from-op (cg/scalar-op :tanh))
          gate (module/comp tanh (mk-affine "gate"))]
      (reify RNNCell
        (cell-model [this] m)
        (output-dim [this] hidden-dim)
        (add-input! [this input last-output last-state]
         (tensors/validate-shape! :lstm-input [input-dim] (:shape input))
         (tensors/validate-shape! :lstm-hidden [hidden-dim] (:shape last-state))
         (let [x (cg/concat 0 input last-output)
               forgot-probs (module/graph forget x)
               keep-probs (module/graph keep x)
               output-probs (module/graph output x)
               state (module/graph gate x)
               ;; combine hadamard of forget past, keep present
               state (cg/+
                      (cg/hadamard forgot-probs last-state)
                      (cg/hadamard keep-probs state))
               output (cg/hadamard output-probs (cg/tanh state))]
           [output state]))))))


(s/defn build-seq
  [cell :- RNNCell inputs]
  (let [factory (-> cell cell-model model/tensor-factory)
        out-dim (output-dim cell)
        zero  (tensors/zeros factory [out-dim])
        init-output (node/constant "h0" factory zero)
        init-state (node/constant "c0" factory  zero)]
    (loop [inputs inputs outputs (list init-output) states (list init-state)]
      (if-let [input (first inputs)]
        (let [last-output (first outputs)
              last-state (first states)
              [output state] (add-input! cell input last-output last-state)]
          (recur (next inputs) (cons output outputs) (cons state states)))
        ;; states/outputs are built in reverse and initial state is
        ;; just so the math works out
        [(->> outputs reverse rest) (->> states reverse rest)]))))
