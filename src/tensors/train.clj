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
   (s/optional-key :learning-rate) s/Num
   (s/optional-key :grad-clip) s/Num})

(def +default-train-opts+
  {:num-iters 100
   :grad-clip 10.0
   :learning-rate 0.01})

(defn reset-batch!
  "Reset all gradients except the target node"
  [loss-node cached-zeros*]
  ;; zero out gradients for parameters and ops nodes
  (doseq [node (graph/post-order-nodes loss-node)
          :when (#{:op :params} (:type node))
          :let [shape (p/safe-get node :shape)
                grad (p/safe-get node :grad)
                factory (p/safe-get node :factory)]]
    (if-let [zs (get-in @cached-zeros* shape)]
      (tensors/copy-from-input! factory grad zs)
      (let [zs (tensors/zeros factory shape)]
        (swap! cached-zeros* assoc shape zs)
        (tensors/copy-from-input! factory grad zs))))
  ;; put a 1.0 on top-level gradient so backward pass
  ;; can propogate non-zero grads backwards
  (tensors/fill!
   (p/safe-get loss-node :factory)
   (p/safe-get loss-node :grad)
   1.0 #_(fn ^double [^shorts _ ^double _] 1.0)))

(defn update-params! [model opts]
  ;; take gradient step
  (doseq [[_ param-node] model]
    (when-let [grad-clip (:grad-clip opts)]
      (tensors/grad-clip!
       (p/safe-get param-node :factory)
       (p/safe-get param-node :grad)
       grad-clip))
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
  (let [total-loss (atom 0.0)
        zeros-cached* (atom {})]
    (doseq [batch (data-gen)]
      (reset-batch! loss-node zeros-cached*)
      (let [batch-loss (run-batch! model loss-node batch)]
        (swap! total-loss + batch-loss))
      (update-params! model opts))
    (let [grads (mapcat (fn [[_ x]] (flatten (tensors/->clj (:factory x) (:grad x)))) model)
          l2-norm (Math/sqrt (reduce (fn [res x] (+ res (* x x))) grads))
          max-l1-norm (apply max (map (fn [x] (Math/abs (double x))) grads))]
      (printf "l2-norm: %.3f\n" l2-norm)
      (printf "max-abs: %.3f\n" max-l1-norm))
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
