(ns tensors.train
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [plumbing.core :as p]
            [tensors.graph :as graph]))

(defn clj-tensor? [x]
  (and (vector? x)
       (or (every? number? x)
           (every? clj-tensor? x))))

(s/defschema DataBatch
  [{s/Str (s/named (s/pred clj-tensor?) "clj-tensor")}])

(s/defschema TrainOpts
  {(s/optional-key :num-iters) s/Int
   (s/optional-key :learning-rate) s/Num})

(def +default-train-opts+
  {:num-iters 10
   :learning-rate 0.01})

(defn reset-batch!
  "Reset all gradients except the target node"
  [loss-node]
  ;; zero out gradients for parameters and ops nodes
  (doseq [node (graph/post-order-nodes loss-node)
          :when (#{:op :params} (:type node))]
    (tensors/fill!
     (p/safe-get node :factory)
     (p/safe-get node :grad)
     (fn ^double [] 0.0)))
  ;; put a 1.0 on top-level gradient so backward pass
  ;; can propogate non-zero grads backwards
  (tensors/fill!
   (p/safe-get loss-node :factory)
   (p/safe-get loss-node :grad)
   (fn ^double [] 1.0)))

(defn update-params! [model opts]
  ;; take gradient step
  (doseq [[_ param-node] model]
    (tensors/grad-step!
     (p/safe-get param-node :factory)
     (p/safe-get param-node :value)
     (p/safe-get opts :learning-rate)
     (p/safe-get param-node :grad))))

(defn run-batch! [model loss-node batch]
  (let [factory (p/safe-get loss-node :factory)]
    (loop [batch-loss 0.0 batch batch]
      (if-let [input->vals (first batch)]
        (let [loss-node (compute/forward-pass! loss-node input->vals)
              loss-val (->> loss-node :value (tensors/->clj factory) first)]
          ;; side-effect to update gradients
          (compute/backward-pass! loss-node)
          (recur (+ batch-loss 0.0 (double loss-val)) (next batch)))
        batch-loss))))

(defn sgd-iter! [model loss-node data-gen opts]
  (let [total-loss (atom 0.0)]
    (doseq [batch (data-gen)]
      (reset-batch! loss-node)
      (let [batch-loss (run-batch! model loss-node batch)]
        (swap! total-loss + batch-loss))
      (update-params! model opts))
    (let [grads (mapcat (fn [[_ x]] (flatten (tensors/->clj (:factory x) (:grad x)))) model)
          l2-norm (Math/sqrt (reduce (fn [res x] (+ res (* x x))) grads))]
      (printf "l2-norm: %.3f\n" l2-norm))
    @total-loss))

(s/defn sgd!
  "`data-gen` should be called to yield a lazy sequence over batches. Each batch
   is a sequence of {input-name clj-tensor} maps (see schema above)"
  ([model :- model/PModel
    target-node :- compute/CompiledRootNode
    data-gen :- (s/=> [DataBatch])
    opts :- TrainOpts]
   (let [opts (merge +default-train-opts+ opts)]
     (dotimes [iter (:num-iters opts)]
       (let [loss (sgd-iter! model target-node data-gen opts)]
         (printf "End of iteration %d: %.3f\n" iter loss)))))
  ([model target-node data-gen]
   (sgd! model target-node data-gen {})))
