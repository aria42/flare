(ns tensors.neanderhal-ops
  (:use [uncomplicate.neanderthal core native])
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [tensors.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.internal.host.mkl :as mkl]
            [schema.core :as s]))


(defn ^:private valid-shape? [shape]
  (= 2 (count shape)))

(defn ^:private ensure-valid-shape?! [shape]
  (when-not (valid-shape? shape)
    (throw (RuntimeException. (str "Must use a matrix shape: " (vec shape))))))


(deftype SumTensorOp []
  compute/TensorOp
  (valid? [this input-nodes]
    (every? #(valid-shape? (:shape %)) input-nodes))
  (forward-pass [this output! inputs]
    (doseq [input inputs]
      (np/axpy! 1.0 input output!)))
  (backward-pass [this _ inputs!]
    (doseq [input! inputs!]
      (np/alter! input! (fn ^double [] 1.0)))))

(deftype MultTensorOp []
  compute/TensorOp)

(deftype SoftMaxTensorOp []
  compute/TensorOp)

(deftype CrossEntropyLossTensorOp []
  compute/TensorOp)

(def ^:private +tensor-ops+
  {:+ (SumTensorOp.)
   :* (MultTensorOp.)
   :soft-max (SoftMaxTensorOp.)
   :cross-entropy-loss (CrossEntropyLossTensorOp.)})

(deftype Factory []
  tensors/PFactory
  (get-op [this op-key]
    (get +tensor-ops+ op-key))
  (from-nums [this nums]
    (let [shape (tensors/guess-shape nums)]
      (case (count shape)
        1 (dv nums)
        2 (let [[row col] shape]
            (dge row col (apply concat nums) {:order :row}))
        ;; else
        (let [err (str "Unallowed shape for neanderthal: " (vec shape))]
          (throw (RuntimeException. err))))))
  (zeros [this shape]
    (case (count shape)
      1 (dv (seq (double-array (first shape))))
      2 (dge (first shape) (second shape))
      (let [err (str "Unallowed shape for neanderthal: " (vec shape))]
        (throw (RuntimeException. err))))))

(def +factory+ (Factory.))
