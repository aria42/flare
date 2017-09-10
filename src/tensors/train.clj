(ns tensors.train
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [plumbing.core :as p]))

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

(defn sgd-iter! [model loss-node data-gen opts]
  (let [factory (p/safe-get loss-node :factory)
        total-loss (atom 0.0)]
    (doseq [batch (data-gen)]
      ;; zero out gradients for each parameter 
      (doseq [[_ param-node] model]
        (tensors/fill!
         (p/safe-get param-node :factory)
         (p/safe-get param-node :grad)
         (fn ^double [] 0.0)))
      ;; run forward-backward on the batch
      ;; gradient accumulates deltas
      (let [batch-loss (atom 0.0)]
        (doseq [input->vals batch]
          (compute/forward-pass! loss-node input->vals)
          (compute/backward-pass! loss-node)
          (let [loss-val (->> loss-node :value (tensors/->clj factory) first)]
            (swap! batch-loss + loss-val)))
        (printf "batch-loss: %.3f" @batch-loss)
        (swap! total-loss + @batch-loss))
      ;; take gradient step
      (doseq [[_ param-node] model]
        (tensors/grad-step!
         (p/safe-get param-node :factory)
         (p/safe-get param-node :value)
         (p/safe-get opts :learning-rate )
         (p/safe-get param-node :grad))))
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
       (let [loss (train-iter! model target-node data-gen opts)]
         (printf "End of iteration %d: %.3f" iter loss)))))
  ([model target-node data-gen]
   (train! model target-node data-gen {})))
