(ns flare.rnn
  (:require [flare.model :as model]
            [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]
            [flare.module :as module])
  (:import [flare.node Node]))

(defprotocol RNNCell
  (cell-model [this]
    "return model underlying cell")
  (output-dim [this]
    "output dimension of cell")
  (input-dim [this]
    "dimension of inputs")
  (init-hidden [this]
    "return initial (output, cell-state) pair")
  (add-input [this input last-hidden]
    "take last hidden and reutrn new hidden"))

(defn lstm-cell
  "Standard LSTM cell 
    https://en.wikipedia.org/wiki/Long_short-term_memory
  without any peepholes or other adaptations.

   The hidden state is a piar (output, cell-state)"
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
      (init-hidden [this] [init-output init-state])
      (add-input [this input [last-output last-state]]
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
  "return sequence of `add-input` outputs for a given `RNNCell`
  and sequence of `inputs`. Can optionally make sequence building
  bidrectional using `bidirectional?` optional third argument.

  Returned sequence will drop the `init-hidden` element which doesn't
  correspond to an input."
  ([cell inputs] (build-seq cell inputs false))
  ([cell inputs bidrectional?]
   (let [factory (-> cell cell-model model/tensor-factory)
         out-dim (output-dim cell)
         hidden (init-hidden cell)
         ;; for bidirectional, concat reversed version of input
         inputs (if bidrectional?
                  (map #(cg/concat 0 %1 %2) inputs (reverse inputs))
                  inputs)]
     (loop [inputs inputs hiddens (list (init-hidden cell))]
       (if-let [input (first inputs)]
         (let [last-hidden (first hiddens)
               hidden (add-input cell input last-hidden)]
           (recur (next inputs) (cons hidden hiddens)))
         ;; states/outputs are built in reverse and initial state is
         ;; just so the math works out
         (reverse (rest hiddens)))))))
