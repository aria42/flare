(ns tensors.rnn
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.core :as tensors]
            [tensors.node :as node]
            [tensors.computation-graph :as cg]
            [tensors.module :as module])
  (:import [tensors.node Node]))

(defprotocol RNNCell
  (cell-model [this])
  (output-dim [this])
  (input-dim [this])
  (add-input! [this input last-output last-state]))

(s/defn lstm-cell
  [m :- model/PModel input-dim :- s/Int hidden-dim :- s/Int]
  (node/with-scope "lstm"
    (let [sigmoid (module/from-op (cg/scalar-op :sigmoid))
          tanh (module/from-op (cg/scalar-op :tanh))
          cat-input-dim (+ input-dim hidden-dim)
          ;; iof (input/output/forget) module
          iof (node/with-scope "iof"
                (module/comp sigmoid
                             (module/affine m (* 3 hidden-dim) [cat-input-dim])))
          gate (node/with-scope "gate"
                 (module/comp tanh
                              (module/affine m hidden-dim [cat-input-dim])))]
      (reify RNNCell
        (cell-model [this] m)
        (output-dim [this] hidden-dim)
        (input-dim [this] input-dim)
        (add-input! [this input last-output last-state]
          (tensors/validate-shape! :lstm-input [input-dim] (:shape input))
          (tensors/validate-shape! :lstm-hidden [hidden-dim] (:shape last-state))
          (let [x (cg/concat 0 input last-output)
                iof-probs (module/graph iof x)
                ;; split iof into (input, forget, output)
                ;; at [0, hidden-dim), [hidden-dim, 2*hidden-dim), [2*hidden-dim,..)
                [input-probs forget-probs output-probs]
                  (cg/split iof-probs 0 hidden-dim (* 2 hidden-dim))
                state (module/graph gate x)
                ;; combine hadamard of forget past, keep present
                state (cg/+
                       (cg/hadamard forget-probs last-state)
                       (cg/hadamard input-probs state))
                output (cg/hadamard output-probs (cg/tanh state))]
            [output state]))))))


(s/defn build-seq
  ([cell inputs] (build-seq cell inputs false))
  ([cell :- RNNCell inputs :- [Node] bidrectional? :- s/Bool]
   (let [factory (-> cell cell-model model/tensor-factory)
         out-dim (output-dim cell)
         zero  (tensors/zeros factory [out-dim])
         init-output (node/constant factory "h0" zero)
         init-state (node/constant factory "c0"  zero)
         ;; for bidirectional, concat reversed version of input
         inputs (if bidrectional?
                  (map #(cg/concat 0 %1 %2) inputs (reverse inputs))
                  inputs)]
     (loop [inputs inputs outputs (list init-output) states (list init-state)]
       (if-let [input (first inputs)]
         (let [last-output (first outputs)
               last-state (first states)
               [output state] (add-input! cell input last-output last-state)]
           (recur (next inputs) (cons output outputs) (cons state states)))
         ;; states/outputs are built in reverse and initial state is
         ;; just so the math works out
         [(->> outputs reverse rest) (->> states reverse rest)])))))
