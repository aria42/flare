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
  (forward-node-pass! [this node]
    (let [output (:value node)
          inputs (mapv :value (:children node))]
      (copy! (first inputs) output)
      (doseq [input (rest inputs)]
        (np/axpy! 1.0 input output)))
    node)
  (backward-node-pass! [this node]
    ;; x = x1 + ... + xn
    ;; dxi/dt = dx/dt * 1
    ;; so just add output grad to inputs
    (let [df_dt (:grad node)
          input-grads (mapv :grad (:children node))]
      (doseq [input-grad input-grads]
        (np/axpy! df_dt input-grad)))
    node))

(defrecord MultTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (doseq [n input-nodes] (ensure-valid-shape?! (:shape n)))
    (when-not (= 2 (count input-nodes))
      (throw (ex-info "Must have two arguments to MultTensorOp"))))
  (forward-node-pass! [this node]
    (let [out (p/safe-get node :value)
          [a b] (mapv #(p/safe-get % :value) (:children node))]
      (scal! 0.0 out)
      (mm! 1.0 a b out))
    node)
  (backward-node-pass! [this node]
    ;; Z[m,n] = X[m,k] Y[k,n]
    ;; dX/dt[m,k] = dZ/dt[m,n] Y^T [n, k]
    (let [dZ (p/safe-get node :grad)
          cs (:children node)
          [X Y] (mapv #(p/safe-get % :value) cs)
          [dX dY] (mapv #(p/safe-get % :grad) cs)]
      ;; update dX
      (np/mm! 1.0 dZ (trans Y) dX)
      ;; update dY
      (np/mm! 1.0 (trans X) dZ dY))
    node))


(defrecord SqueezeTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (when-not (= 2 (count (:shape (first input-nodes))))
      (throw (ex-info "Need matrix shape"
                      {:shape (:shape (first input-nodes))}))))
  (forward-node-pass! [this node]
    (let [output (:value node)
          input (-> node :children first :value)]
      (copy! (view-vctr input) output))
    node)
  (backward-node-pass! [this node]
    (let [out-grad (:grad node)
          in-grad (-> node :children first :grad)]
      (copy! out-grad (view-vctr in-grad)))
    node))

(defrecord StrechTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes]
    (let [shape (:shape (first input-nodes))]
      (when-not (tensors/vector-shape? shape)
        (throw (ex-info "Need vector shape" {:shape shape})))))
  (forward-node-pass! [this node]
    (let [output (:value node)
          input (-> node :children first :value)]
      (copy! input (view-vctr output)))
    node)
  (backward-node-pass! [this node]
    (let [out-grad (:grad node)
          in-grad (-> node :children first :grad)]
      (axpy! out-grad (view-ge in-grad)))
    node))

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
  (forward-node-pass! [this node]
    (let [[activations-node label-node] (:children node)
          activations (-> activations-node :value)
          probs (soft-max activations)
          label (-> label-node :value (entry 0) long)
          correct-prob (real/entry probs label)]
      (when (>= label (dim probs))
        (throw (ex-info "Label index out-of-bounds"
                        {:label label
                         :dim (dim activations)})))
      (alter! (:value node) 0
              (fn ^double [^double _]
                (- (Math/log correct-prob))))
      (assoc node ::probs probs)))
  (backward-node-pass! [this node]
    (let [[activations-node label-node] (:children node)
          probs (::probs node)
          loss-grad-val (-> node :grad (real/entry 0))]
      ;; l = (i == label) log(pi)
      (when-let [g (:grad label-node)]
        (throw (ex-info "Don't support label differentiation")))
      ;; d pi / dt = dl/dt (1.0/pi)
      (let [gold-idx (long (first (:value label-node)))
            activations (:value activations-node)]
        (alter! (:grad activations-node)
                (fn ^double [^long idx ^double _]
                  (* loss-grad-val
                     (- (real/entry probs idx)
                        (if (= idx gold-idx) 1.0 0.0)))))))
    node))

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
