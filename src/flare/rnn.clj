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
      [;; stack (input, output, forget, gate) params
       ;; there is x_t -> h_t and h_{t-1} -> h_t transforms
       ;; input->hidden (module/affine model (* 4 hidden-dim) [input-dim])
       input->output (module/affine model hidden-dim [input-dim])
       input->forget (module/affine model hidden-dim [input-dim])
       input->input (module/affine model hidden-dim [input-dim])
       input->state  (module/affine model hidden-dim [input-dim])
       prev->output (module/affine model hidden-dim [hidden-dim])
       prev->forget (module/affine model hidden-dim [hidden-dim])
       prev->input (module/affine model hidden-dim [hidden-dim])
       prev->state  (module/affine model hidden-dim [hidden-dim])
       ;;prev->hidden (module/affine model (* 4 hidden-dim) [hidden-dim])
       factory (model/tensor-factory model)
       zero  (flare/zeros factory [hidden-dim])
       init-output (node/const factory "h0" zero)
       init-state (node/const factory "c0"  zero)]
    (reify RNNCell
      (cell-model [this] model)
      (output-dim [this] hidden-dim)
      (input-dim [this] input-dim)
      (init-pair [this] [init-output init-state])
      (add-input! [this input last-output last-state]
        (flare/validate-shape! [input-dim] (:shape input))
        (flare/validate-shape! [hidden-dim] (:shape last-state))
        (let [gate-acts (cg/+ (module/graph prev->state last-output)
                              (module/graph input->state input))
              keep-acts (cg/+ (module/graph prev->input last-output)
                              (module/graph input->input input))
              forget-acts (cg/+ (module/graph prev->forget last-output)
                                (module/graph input->forget input))
              output-acts (cg/+
                           (module/graph input->output input)
                           (module/graph prev->output last-output))
              state gate-acts
              state 
              (cg/+
               (cg/hadamard (cg/sigmoid forget-acts) last-state)
               (cg/hadamard (cg/sigmoid keep-acts) state))
              output-acts (module/graph input->output input)
              output (cg/hadamard (cg/tanh state) (cg/sigmoid output-acts))]
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
