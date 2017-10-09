(ns tensors.rnn
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.core :as tensors]
            [tensors.node :as node]
            [tensors.computation-graph :as cg]))

(defprotocol RNNCell
  (cell-model [this])
  (output-dim [this])
  (add-input! [this input last-output last-state]))

(defn -affine-transform
  [scope-prefix model input-dim hidden-dim]
  (node/with-scope scope-prefix
    (let [W-shape [hidden-dim (+ hidden-dim input-dim)]
          W (model/add-params! model W-shape :name "W")
          b (model/add-params! model [hidden-dim] :name "b")]
      (fn [state]
        (cg/+ (cg/* W state) b)))))

(s/defn lstm-cell
  [m :- model/PModel input-dim :- s/Int hidden-dim :- s/Int]
  (node/with-scope "lstm"
    (let [affine (fn [prefix]
                   (-affine-transform prefix m input-dim hidden-dim))
          forget-fn (comp cg/sigmoid (affine "forget"))
          keep-fn (comp cg/sigmoid (affine "keep"))
          out-fn (comp cg/sigmoid (affine "out"))
          trans-fn (comp cg/tanh (affine "trans"))]
      (reify RNNCell
        (cell-model [this] m)
        (output-dim [this] hidden-dim)
        (add-input! [this input last-output last-state]
         (tensors/validate-shape! :lstm-input [input-dim] (:shape input))
         (tensors/validate-shape! :lstm-hidden [hidden-dim] (:shape last-output))
         (let [x (cg/concat 0 last-output input)
               forgot-probs (forget-fn x)
               keep-probs (keep-fn x)
               out-probs (out-fn x)
               state (trans-fn x)
               ;; combine hadamard of forget past, keep present
               state (cg/+
                           (cg/hadamard forgot-probs last-state)
                           (cg/hadamard keep-probs state))
               output (cg/hadamard out-probs (cg/tanh state))]
           [output state]))))))


(s/defn build-seq
  [cell :- RNNCell inputs]
  (let [factory (-> cell cell-model model/tensor-factory)
        out-dim (output-dim cell)
        zero  (tensors/zeros factory [out-dim])
        init-output (node/constant "h0" [out-dim] zero)
        init-state (node/constant "c0" [out-dim]  zero)]
    (loop [inputs inputs outputs (list init-output) states (list init-state)]
      (if-let [input (first inputs)]
        (let [last-output (first outputs)
              last-state (first states)
              [output state] (add-input! cell input last-output last-state)]
          (recur (next inputs) (cons output outputs) (cons state states)))
        ;; states/outputs are built in reverse and initial state is
        ;; just so the math works out
        [(->> outputs reverse (drop 1)) (->> states reverse (drop 1))]))))
