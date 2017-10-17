(ns tensors.train
  (:require [schema.core :as s]
            [tensors.model :as model]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [tensors.report :as report]
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
   (s/optional-key :batch-reporter) report/Reporter
   (s/optional-key :grad-clip) s/Num})

(def +default-train-opts+
  {:num-iters 100
   :grad-clip 10.0
   :learning-rate 0.01})

(defn reset-batch!
  "Reset all gradients except the target node"
  [model]
  ;; zero out gradients for param nodes
  (let [factory (model/tensor-factory model)]
    (doseq [[_ param-node] model]
      (tensors/fill! factory (p/safe-get param-node :grad) 0.0))))

(defn ^:static clip ^double [^double x ^double min ^double max]
  (if (> x max)
    max
    (if (< x min)
      min
      x)))

(defn update-params! [model batch opts]
  ;; take gradient step
  (let [factory (model/tensor-factory model)]
    (doseq [[_ param-node] model]
      (let [grad-clip (double (:grad-clip opts 10.00))
            normalizer (/ 1.0 (double (count batch)))
            clip-min (- (Math/abs grad-clip))
            clip-max (Math/abs grad-clip)]
        (tensors/fill!
         factory
         (p/safe-get param-node :grad)
         (fn ^double [^longs _ ^double x]
           (clip (* normalizer x) clip-min clip-max))))
      (tensors/grad-step!
       factory
       (p/safe-get param-node :value)
       (p/safe-get opts :learning-rate)
       (p/safe-get param-node :grad)))))

(defn run-batch! [model get-loss-node batch]
  (let [factory (model/tensor-factory model)]
    (reset-batch! model)
    (loop [batch-loss 0.0 batch batch]
      (if-let [data (first batch)]
        (if-let [loss-node (get-loss-node data)]
          (let [_ (tensors/copy-from-input! factory (:grad loss-node) [1.0])
                loss-val (->> loss-node :value (tensors/->clj factory) first)]
            ;; side-effect to update gradients
            (compute/backward-pass! loss-node)
            (recur (+ batch-loss 0.0 (double loss-val)) (next batch)))
          (recur batch-loss (next batch)))
        batch-loss))))

(defn sgd-iter! [model get-loss-node data-gen opts]
  (let [total-loss (atom 0.0)
        factory (model/tensor-factory model)]
    (doseq [batch (data-gen)]
      (let [batch-loss (run-batch! model get-loss-node batch)
            batch-info {:batch-loss batch-loss :model model :batch batch}]
        (when-let [reporter (:batch-reporter opts)]
          (report/update! reporter batch-info)
          (when-let [r (report/gen reporter)]
            (clojure.pprint/pprint r)))
        (swap! total-loss + batch-loss)
        (when-let [callback (:batch-callback opts)]
          (when-let [report (callback model batch batch-loss)]
            (println (clojure.pprint/pprint report)))))
      (update-params! model batch opts))
    (when-let [reporter (:iter-reporter )])
    @total-loss))

(s/defn sgd!
  "`data-gen` should be called to yield a lazy sequence over batches. Each batch
   is a sequence of {input-name clj-tensor} maps (see schema above)"
  ([model :- model/PModel
    get-loss-node
    data-gen :- (s/=> [DataBatch])
    opts :- TrainOpts]
   (let [factory (model/tensor-factory model)
         opts (merge +default-train-opts+ opts)]
     (dotimes [iter (:num-iters opts)]
       (when-let [reporter (:batch-reporter opts)]
         (report/clear! reporter))
       (when-let [reporter (:iter-reporter opts)]
         (report/clear! reporter))
       (printf "Iteration %d\n" iter)
       (let [time (System/currentTimeMillis)
             loss (sgd-iter! model get-loss-node data-gen opts)]
         (let [delta-ms (- (System/currentTimeMillis) time)]
           (printf "End of iteration %d: %.3f (%d ms) \n" iter loss delta-ms)
           (.flush System/out))))))
  ([model get-loss-node data-gen]
   (sgd! model get-loss-node data-gen {})))


(defn static-graph-sgd!
  [model loss-node data-gen opts]
  (sgd!
   model
   (fn [input->vals] (compute/forward-pass! loss-node (model/tensor-factory model) input->vals))
   data-gen
   opts))
