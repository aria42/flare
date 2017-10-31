(ns tensors.train
  (:require
   [clojure.pprint :as pprint]
   [schema.core :as s]
   [tensors.model :as model]
   [tensors.compute :as compute]
   [tensors.core :as tensors]
   [tensors.report :as report]
   [tensors.optimize :as optimize]
   [plumbing.core :as p]
   [tensors.graph :as graph]
   [tensors.module :as module]))

(s/defschema TrainOpts
  {(s/optional-key :num-iters) s/Int
   (s/optional-key :optimizer) optimize/Optimizer
   (s/optional-key :learning-rate) s/Num
   (s/optional-key :iter-reporter) [report/Reporter]})

(def +default-train-opts+
  {:num-iters 100
   :learning-rate 0.01})

(defn run-batch!
  "run forward/backward passes for each of the loss graphs created
   by `loss-module` on the batch.

   Each entry in batch as treated as arguments to `module/graph`
   to build an expression for example loss"
  [factory get-loss-node batch cache]
  (loop [batch-loss 0.0 batch batch]
    (if-let [data (first batch)]
      (if-let [loss-node (get-loss-node data)]
        (let [loss-node (compute/forward-pass! factory loss-node cache)
              _ (tensors/copy! factory (:grad loss-node) [1.0])
              loss-val (-> loss-node :value seq first)]
          ;; side-effect to update gradients
          (compute/backward-pass! factory loss-node cache)
          (recur (+ batch-loss 0.0 (double loss-val)) (next batch)))
        (recur batch-loss (next batch)))
      batch-loss)))

(defn iter! [model optimizer get-loss-node data-gen opts]
  (let [total-loss (atom 0.0)
        factory (model/tensor-factory model)
        cache (compute/cache factory 100)]
    (doseq [batch (data-gen)]
      (optimize/reset-batch! optimizer model)
      (let [batch-loss (run-batch! factory get-loss-node batch cache)]
        (swap! total-loss + batch-loss))
      (optimize/update-model! optimizer model))
    @total-loss))

(s/defn train!
  "`data-gen` should be called to yield a lazy sequence over batches. Each batch
   is a sequence of {input-name clj-tensor} maps (see schema above)"
  ([model :- model/PModel
    get-loss-node 
    data-batch-fn
    opts :- TrainOpts]
   (let [factory (model/tensor-factory model)
         optimizer (:optimizer opts (optimize/->Adadelta factory 1.0 0.9 1e-6))
         opts (merge +default-train-opts+ opts)]
     (println "Optimizing with" (type optimizer))
     (optimize/init-model! optimizer model)
     (dotimes [iter (:num-iters opts)]
       (doseq [reporter (:iter-reporter opts)]
         (report/clear! reporter))
       (printf "Iteration %d\n" iter)
       (let [time (System/currentTimeMillis)
             loss (iter! model optimizer get-loss-node data-batch-fn opts)]
         (doseq [reporter (:iter-reporter opts)]
           (when-let [r (report/gen reporter)]
             (pprint/pprint r)))
         (let [delta-ms (- (System/currentTimeMillis) time)]
           (printf "End of iteration %d: %.3f (%d ms) \n" iter loss delta-ms)
           (.flush System/out))))))
  ([model get-loss-node data-batch-fn]
   (train! model get-loss-node data-batch-fn {})))
