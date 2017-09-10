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
  (let [factory (p/safe-get loss-node :factory)]
    (doseq [batch (data-gen)]
      ;; zero out gradients for each parameter 
      (doseq [[_ param-node] model]
        (tensors/fill!
         (p/safe-get param-node :factory)
         (p/safe-get param-node :grad)
         (fn ^double [] 0.0)))
      ;; run forward-backward on the batch
      ;; gradient accumulates deltas
      (doseq [input->vals batch]
        (compute/forward-pass! loss-node input->vals)
        (let [loss-val (->> loss-node :value (tensors/->clj factory) first)]
          (printf "loss-val %0.5f" loss-val))
        (compute/backward-pass! loss-node))
      ;; take gradient step
      (doseq [[_ param-node] model]
        (tensors/grad-step!
         (p/safe-get param-node :factory)
         (p/safe-get param-node :value)
         (p/safe-get opts :learning-rate )
         (p/safe-get param-node :grad))))))

(s/defn sgd!
  "`data-gen` should be called to yield a lazy sequence over batches. Each batch
   is a sequence of {input-name clj-tensor} maps (see schema above)"
  ([model :- model/PModel
    target-node :- compute/CompiledRootNode
    data-gen :- (s/=> [DataBatch])
    opts :- TrainOpts]
   (let [opts (merge +default-train-opts+ opts)]
     (dotimes [iter (:num-iters opts)]
       (train-iter! model target-node data-gen opts))))
  ([model target-node data-gen]
   (train! model target-node data-gen {})))
