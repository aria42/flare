(ns tensors.nd4j-ops
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.ndarray INDArray])
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [tensors.compute :as compute]
            [uncomplicate.neanderthal.core :as np]
            [uncomplicate.neanderthal.native :as native]
            [uncomplicate.neanderthal.real :as real]
            [schema.core :as s]
            [plumbing.core :as p]))


(set! *unchecked-math* :warn-on-boxed)

(defrecord SumTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes] true)
  (forward-node-pass! [this node]
    (let [^INDArray output (:value node)
          inputs (mapv :value (:children node))]
      (.add output ^INDArray (first inputs))
      (doseq [^INDArray other (rest inputs)]
        (.addi output other)))
    node)
  (prep [this node]
    node)
  (backward-node-pass! [this node]
    ;; x = x1 + ... + xn
    ;; dxi/dt = dx/dt * 1
    ;; so just add output grad to inputs
    (let [^INDArray df_dt (:grad node)
          input-grads (mapv :grad (:children node))]
      (doseq [input-grad input-grads :when input-grad]
        (.assign ^INDArray input-grad df_dt)))
    node))

(def ^:private +tensor-ops+
  {:+ ->SumTensorOp
   :* nil
   :squeeze nil
   :strech nil
   :cross-entropy-loss nil})

(defrecord Factory []
  tensors/PFactory
  (get-op [this op-key]
    ((get +tensor-ops+ op-key)))
  (->clj [this tensor]
    ;; horrible horrible hack
    (read-string (.replace (.toString tensor) "\n" "")))
  (fill! [this tensor get-val-fn]
    )
    (from-nums [this nums]
  )
    (grad-step! [this weights alpha grad]
  )
    (copy-from-input! [this tensor! nums]
  )
    (zeros [this shape]
  ))


(comment 
  (defrecord MultTensorOp []
    compute/TensorOp
    (ensure-valid?! [this input-nodes]
      (doseq [n input-nodes] (ensure-valid-shape?! (:shape n)))
      (when-not (= 2 (count input-nodes))
        (throw (ex-info "Must have two arguments to MultTensorOp"))))
    (prep [this node]
      node)
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
        (when dX
          (np/mm! 1.0 dZ (trans Y) dX))
        ;; update dY
        (when dY
          (np/mm! 1.0 (trans X) dZ dY)))
      node))


  (defrecord SqueezeTensorOp []
    compute/TensorOp
    (ensure-valid?! [this input-nodes]
      (when-not (= 2 (count (:shape (first input-nodes))))
        (throw (ex-info "Need matrix shape"
                        {:shape (:shape (first input-nodes))}))))
    (prep [this node]
      node)
    (forward-node-pass! [this node]
      (let [output (:value node)
            input (-> node :children first :value)]
        (copy! (view-vctr input) output))
      node)
    (backward-node-pass! [this node]
      (let [out-grad (:grad node)
            in-grad (-> node :children first :grad)]
        (when in-grad
          (copy! out-grad (view-vctr in-grad))))
      node))

  (defrecord StrechTensorOp []
    compute/TensorOp
    (ensure-valid?! [this input-nodes]
      (let [shape (:shape (first input-nodes))]
        (when-not (tensors/vector-shape? shape)
          (throw (ex-info "Need vector shape" {:shape shape})))))
    (prep [this node]
      node)
    (forward-node-pass! [this node]
      (let [output (:value node)
            input (-> node :children first :value)]
        (copy! input (view-vctr output)))
      node)
    (backward-node-pass! [this node]
      (let [out-grad (:grad node)
            in-grad (-> node :children first :grad)]
        (when in-grad
          (axpy! out-grad (view-ge in-grad))))
      node))

  (defn soft-max! [scores probs!]
    ;; copy scores + exp in place
    (alter! probs!
            (fn ^double [^long idx ^double x] 
              (Math/exp (real/entry scores idx))))
    ;; normalize
    (let [Z (double (asum probs!))]
      (scal! (/ 1.0 Z) probs!)
      probs!))

  (defrecord CrossEntropyLossTensorOp []
    compute/TensorOp
    (ensure-valid?! [this [activations label]]
      true)
    (prep [this node]
      (let [[activations-node label-node] (:children node)
            len (-> activations-node :shape first)]
        (assoc node ::probs (dv len))))
    (forward-node-pass! [this node]
      (let [[activations-node label-node] (:children node)
            activations (-> activations-node :value)
            probs (soft-max! activations (p/safe-get node ::probs))
            label (-> label-node :value (entry 0) long)
            correct-prob (real/entry probs label)]
        (when (>= label (dim probs))
          (throw (ex-info "Label index out-of-bounds"
                          {:label label
                           :dim (dim activations)})))
        (real/entry! (:value node) 0 (- (Math/log correct-prob)))
        node))
    (backward-node-pass! [this node]
      (let [[activations-node label-node] (:children node)
            probs (p/safe-get node ::probs)
            loss-grad-val (-> node :grad (real/entry 0))]
        ;; l = (i == label) log(pi)
        (when-let [g (:grad label-node)]
          (throw (ex-info "Don't support label differentiation")))
        ;; d pi / dt = dl/dt (1.0/pi)
        (let [gold-idx (long (first (:value label-node)))
              activations (:value activations-node)]
          (when-let [act-grad (:grad activations-node)]
            (alter! act-grad
                    (fn ^double [^long idx ^double _]
                      (* loss-grad-val
                         (- (real/entry probs idx)
                            (if (= idx gold-idx) 1.0 0.0))))))))
      node))

  )
