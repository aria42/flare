(ns flare.rnn
  (:require [flare.model :as model]
            [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]
            [flare.module :as module])
  (:import [flare.node Node]))

(defprotocol RNNCell 
  (cell-model [this])
  (output-dim [this])
  (input-dim [this])
  (init-pair [this])
  (add-input! [this input last-output last-state]))

(defn lstm-cell
  "Standard LSTM cell 
    https://en.wikipedia.org/wiki/Long_short-term_memory
  without any peepholes or other adaptations."
  [model ^long input-dim ^long hidden-dim]
  (node/let-scope
      [;; concatenate previous output and cur input
       cat-dim (+ input-dim hidden-dim)
       ;; stack (input, output, forget, state) params
       ;; one affine module W_(i,o,f,s) * x_(prev, input) + b_(i,o,f,s)
       input->gates (module/affine model (* 4 hidden-dim) [cat-dim])
       zero  (flare/zeros [hidden-dim])
       init-output (node/const "h0" zero)
       init-state (node/const "c0"  zero)]
    (reify RNNCell
      (cell-model [this] model)
      (output-dim [this] hidden-dim)
      (input-dim [this] input-dim)
      (init-pair [this] [init-output init-state])
      (add-input! [this input last-output last-state]
        (flare/validate-shape! [input-dim] (:shape input))
        (flare/validate-shape! [hidden-dim] (:shape last-state))
        (let [x (cg/concat 0 input last-output)
              gates (module/graph input->gates x)
              ;; split (i,o,f) and state
              [iof, state] (cg/split gates 0 (* 3 hidden-dim))
              ;; one big sigmloid then split into (input, forget, output)
              [input-probs forget-probs output-probs]
                (cg/split (cg/sigmoid iof) 0 hidden-dim (* 2 hidden-dim))
              ;; combine hadamard of forget past, keep present
              state (cg/+
                     (cg/hadamard forget-probs last-state)
                     (cg/hadamard input-probs (cg/tanh state)))
              output (cg/hadamard output-probs (cg/tanh state))]
          [output state])))))


(defn build-seq
  "return `[outputs states]` pair where the `outputs` and `states`
  correspond to the output of the `RNNCell` for param `cell` on `inputs`
  and `states` represents the sequence of hidden states of the cell
  on each input."
  ([cell inputs] (build-seq cell inputs false))
  ([cell inputs bidrectional?]
   (let [factory (-> cell cell-model model/tensor-factory)
         out-dim (output-dim cell)
         [init-output init-state] (init-pair cell)
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
