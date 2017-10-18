(ns tensors.nd4j-ops
  (:import [org.nd4j.linalg.factory Nd4j]
           [org.nd4j.linalg.api.iter NdIndexIterator]
           [org.nd4j.linalg.ops.transforms Transforms]
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
    (let [^INDArray output (p/safe-get node :value)
          inputs (mapv :value (:children node))]
      (.assign output ^INDArray (first inputs))
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

(defrecord MultTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes] true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [^INDArray out (p/safe-get node :value)
          [a b] (mapv #(p/safe-get % :value) (:children node))]
      (.assign out (.mmul ^INDArray a ^INDArray b)))
    node)
  (backward-node-pass! [this node]
    ;; Z[m,n] = X[m,k] Y[k,n]
    ;; dX/dt[m,k] = dZ/dt[m,n] Y^T [n, k]
    (let [^INDArray dZ (p/safe-get node :grad)
          cs (:children node)
          [^INDArray X ^INDArray Y] (mapv #(p/safe-get % :value) cs)
          [^INDArray dX ^INDArray dY] (mapv #(p/safe-get % :grad) cs)]
      ;; update dX
      (when dX
        (let [update (.mmul dZ (.transpose Y))]
          (.addi dX update)))
      ;; update dY
      (when dY
        (let [update (.mmul (.transpose X) dZ)]
          (.addi dY update))))
    node))

(defrecord SqueezeTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes] true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [^INDArray output (:value node)
          ^INDArray input (-> node :children first :value)]
      (.assign output input))
    node)
  (backward-node-pass! [this node]
    (let [^INDArray out-grad (:grad node)
          ^INDArray in-grad (-> node :children first :grad)]
      (when in-grad
        (.assign in-grad out-grad)))
    node))

(defrecord StrechTensorOp []
  compute/TensorOp
  (ensure-valid?! [this input-nodes] true)
  (prep [this node] node)
  (forward-node-pass! [this node]
    (let [^INDArray output (:value node)
          ^INDArray  input (-> node :children first :value)]
      (.assign output input))
    node)
  (backward-node-pass! [this node]
    (let [^INDArray out-grad (:grad node)
          ^INDArray in-grad (-> node :children first :grad)]
      (when in-grad
        (.assign in-grad out-grad)))
    node))

(s/defn soft-max! :- INDArray
  [scores :- INDArray probs! :- INDArray]
  ;; copy scores + exp in place
  (.assign probs! scores)
  (Transforms/exp probs! false)
  ;; normalize
  (let [Z (.sumNumber probs!)]
    (.muli probs! (Double. (/ 1.0 (.doubleValue Z))))
    probs!))

(defrecord CrossEntropyLossTensorOp []
    compute/TensorOp
    (ensure-valid?! [this [activations label]]
      true)
    (prep [this node]
      (let [[activations-node label-node] (:children node)
            len (-> activations-node :shape first int)]
        (assoc node ::probs (Nd4j/zeros len))))
    (forward-node-pass! [this node]
      (let [[activations-node label-node] (:children node)
            ^INDArray activations (p/safe-get activations-node :value)
            probs (soft-max! activations (p/safe-get node ::probs))
            ^INDArray label (p/safe-get label-node :value)
            label-value (long (.getDouble label 0))
            correct-prob (.getDouble probs label-value)]
        (when (>= label-value (.length probs))
          (throw (ex-info "Label index out-of-bounds"
                          {:label label
                           :dim (.size activations 0)})))
        (.putScalar
         ^INDArray (p/safe-get node :value)
         (int 0)
         (- (Math/log correct-prob)))
        node))
    (backward-node-pass! [this node]
      (let [[activations-node label-node] (:children node)
            ^INDArray probs (p/safe-get node ::probs)
            ^INDArray loss-grad (p/safe-get node :grad)
            loss-grad-val (.getDouble loss-grad 0)]
        ;; l = (i == label) log(pi)
        (when-let [g (:grad label-node)]
          (throw (ex-info "Don't support label differentiation")))
        ;; d pi / dt = dl/dt (1.0/pi)
        (let [^INDArray label (p/safe-get label-node :value)
              gold-idx (int (.getDouble label 0))
              ^INDArray activations (p/safe-get activations-node :value)]
          (when-let [^INDArray act-grad (p/safe-get activations-node :grad)]
            (let [n (.length act-grad)]
              (dotimes [idx n]
                (.putScalar
                 act-grad
                 (int idx)
                 (* loss-grad-val
                    (- (.getDouble probs idx)
                       (if (= idx gold-idx) 1.0 0.0)))))))))
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
    ;; horrible horrible hack
    (let [val (read-string (.replace (.toString tensor) "\n" ""))]
      (if (number? val)
        [val]
        val)))
  (transform! [this tensor get-val-fn]
    (let [^INDArray tensor tensor]
      (if (number? get-val-fn)
        (.assign tensor ^Number get-val-fn)
        (let [iter (NdIndexIterator. (.ordering tensor) (.shape tensor))]
          (doseq [^ints idx-path (iterator-seq iter)]
            (let [old-val (.getDouble tensor idx-path)
                  val-fn ^clojure.lang.IFn$ODD get-val-fn
                  new-val (.invokePrim val-fn (long-array idx-path) old-val)]
              (.putScalar tensor idx-path new-val)))))
      tensor))
  (from-nums [this nums]
    (Nd4j/create (double-array (flatten nums))
                 (int-array (tensors/guess-shape nums))
                 \c))
  (grad-step! [this weights alpha grad]
    (let [update (.mul ^INDArray grad ^Double alpha)]
      (.addi ^INDArray grad update)))
  (copy-from-input! [this tensor! nums]
    (if (instance? INDArray nums)
      (.assign ^INDArray tensor! ^INDArray nums)
      (.assign ^INDArray tensor! ^INDArray (tensors/from-nums this nums))))
  (zeros [this shape]
    (Nd4j/zeros (int-array shape))))
