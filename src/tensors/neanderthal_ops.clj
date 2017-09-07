(ns tensors.neanderthal-ops
  (:use [uncomplicate.neanderthal core native])
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [tensors.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as real]
            [schema.core :as s]
            [plumbing.core :as p]))


(defn ^:private valid-shape? [shape]
  (<= (count shape) 2))

(defn ^:private ensure-valid-shape?! [shape]
  (when-not (valid-shape? shape)
    (throw (ex-info "Must use a vector (1d) or matrix (2d) shape"
                    {:shape shape}))))

(defrecord SumTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [n input-nodes] (ensure-valid-shape?! (:shape n))))
  (forward-node-pass! [this output! inputs]
    (copy! (:value (first inputs)) (:value output!))
    (doseq [input (rest inputs)]
      (np/axpy! 1.0 (:value input) (:value output!)))
    output!)
  (backward-node-pass! [this output inputs!]
    ;; x = x1 + ... + xn
    ;; dxi/dt = dx/dt * 1
    ;; so just add output grad to inputs
    (doseq [input! inputs!]
      (np/axpy! (:grad output) (:grad inputs!)))
    output))

(defrecord MultTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [n input-nodes] (ensure-valid-shape?! (:shape n)))
    (when-not (= 2 (count input-nodes))
      (throw (ex-info "Must have two arguments to MultTensorOp"))))
  (forward-node-pass! [this output! inputs]
    (let [out (p/safe-get output! :value)
          [a b] (map #(p/safe-get % :value) inputs)]
      (scal! 0.0 out)
      (mm! 1.0 a b out))
    output!)
  (backward-node-pass! [this output inputs!]
    ;; Z[m,n] = X[m,k] Y[k,n]
    ;; dX/dt[m,k] = dZ/dt[m,n] Y^T [n, k]
    (let [dZ (p/safe-get output :grad)
          [X Y] (map #(p/safe-get % :value) inputs!)
          [dX dY] (map #(p/safe-get % :grad) inputs!)]
      ;; update dX
      (np/mm! 1.0 dZ 1.0 (trans Y) dX)
      ;; update dY
      (np/mm! 1.0 dZ 1.0 (trans X) dY))
    output))


(defrecord SqueezeTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (when-not (= 2 (count (:shape (first input-nodes))))
      (throw (ex-info "Need matrix shape"
                      {:shape (:shape (first input-nodes))}))))
  (forward-node-pass! [this output! [input]]
    (copy! (view-vctr (:value input)) (:value output!))
    output!)
  (backward-node-pass! [this output inputs!]
    (copy! (view-vctr (:grad output)) (:grad (first inputs!)))
    output))

(defrecord StrechTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (let [shape (:shape (first input-nodes))]
      (when-not (tensors/vector-shape? shape)
        (throw (ex-info "Need vector shape" {:shape shape})))))
  (forward-node-pass! [this output! [input]]
    (let [strech-graph-op (:graph-op output!)
          dim-to-insert (:dim-to-insert strech-graph-op)]
      (if (= dim-to-insert 1)
        (copy! (view-ge (:value input)) (:value output!))
        (copy! (trans (view-ge (:value input))) (:value output!)))
      output!))
  (backward-node-pass! [this output [input!]]
    (let [strech-graph-op (:graph-op output)
          dim-to-insert (:dim-to-insert strech-graph-op)]
      (if (= dim-to-insert 1)
        (axpy! (:grad output) (view-ge (:grad input!)))
        (axpy! (:grad output) (view-ge (:grad input!)))))
    output))

(defn soft-max [scores]
  (let [probs (copy scores)]
    ;; exp in place
    (alter! probs (fn ^double [^double x] (Math/exp x)))
    ;; normalize
    (let [Z (double (asum probs))]
      (scal! (/ 1.0 Z) probs)
      probs)))

(defrecord CrossEntropyLossTensorOp []
  compute/TensorOp
  (ensure-valid?! [this [activations label]]
    true)
  (forward-node-pass! [this output! [activations-node label-node]]
    (let [activations (-> activations-node :value)
          probs (soft-max activations)
          label (-> label-node :value (entry 0) long)
          correct-prob (real/entry probs label)]
      (when (>= label (dim probs))
        (throw (ex-info "Label index out-of-bounds"
                        {:label label
                         :dim (dim activations)})))
      (alter! (:value output!) 0
              (fn ^double [^double _]
                (Math/log correct-prob)))))
  (backward-node-pass! [this output [activations-node! label-node]]
    ;; l = (i == label) log(pi)
    (when-let [g (:grad label-node)]
      (throw (ex-info "Don't support label differentiation")))
    ;; d pi / dt = dl/dt (1.0/pi)
    (let [gold-idx (long (first (:value label-node)))
          activations (:value activations-node!)]
      (alter! (:grad activations-node!) gold-idx
              (fn ^double [^long idx]
                (/ (real/entry (:grad output) idx)
                   (real/entry activations idx)))))))

(def ^:private +tensor-ops+
  {:+ ->SumTensorOp
   :* ->MultTensorOp
   :squeeze ->SqueezeTensorOp
   :strech ->StrechTensorOp
   :cross-entropy-loss ->CrossEntropyLossTensorOp})

(defrecord Factory []
  tensors/PFactory
  (get-op [this op-key]
    ((get +tensor-ops+ op-key)))
  (->clj [this tensor]
    (if (vctr? tensor)
      (seq tensor)
      (doall (map seq (rows tensor)))))
  (from-nums [this nums]
    (let [shape (tensors/guess-shape nums)]
      (case (count shape)
        1 (dv nums)
        2 (let [[row col] shape]
            (dge row col (apply concat nums) {:order :row}))
        ;; else
        (let [err (str "Unallowed shape for neanderthal: " (vec shape))]
          (throw (RuntimeException. err))))))
  (copy-from-input! [this tensor! nums]
    (copy! (tensors/from-nums this nums) tensor!))
  (zeros [this shape]
    (case (count shape)
      1 (dv (seq (double-array (first shape))))
      2 (dge (first shape) (second shape))
      (let [err (str "Unallowed shape for neanderthal: " (vec shape))]
        (throw (RuntimeException. err))))))
