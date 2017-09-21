(ns tensors.rnn)

(defprotocol RNNCell
  (add-input! [this input]))
