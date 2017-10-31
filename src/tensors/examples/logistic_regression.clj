(set! *unchecked-math* true)
(ns tensors.examples.logistic-regression
  (:gen-class)
  (:require [tensors.neanderthal-ops :as no]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [tensors.model :as model]
            [tensors.module :as module]
            [tensors.train :as train]
            [tensors.computation-graph :as cg]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.tools.cli :refer [parse-opts]]
            [tensors.node :as node]
            [tensors.report :as report]))

(tensors/set-factory! (no/factory))

(defn data-generator [num-classes num-feats]
  (let [r (java.util.Random. 0)
        W (dge num-classes num-feats)
        b (dv num-classes)]
    (alter! W (fn ^double [^long idx1 ^long idx2 ^double w] (.nextDouble r)))
    (alter! b (fn ^double [^long idx ^double w] (.nextDouble r)))
    (fn []
      (let [rand-feats (mapv (fn [_] (if (.nextBoolean r) 1.0 0.0)) (range num-feats))
            f (dv rand-feats)
            class-idx (.nextInt r (int num-classes))
            activations (axpy (mv W f) b)
            label (imax activations)]
        {"f" f "label" (dv [label])}))))

(defn logistic-regression [model num-classes num-feats]
  (let [feat-vec (node/input "f" [num-feats])
        label (node/input "label" [1])
        aff (module/affine model num-classes [num-feats])
        activations (module/graph aff feat-vec)
        loss (cg/cross-entropy-loss activations label)]
    [loss activations]))

(defn train [{:keys [num-examples, num-iters, num-feats, num-batch] :as opts}]
  (println "options " opts)
  (let [num-classes 5
        factory (no/factory)
        m (model/simple-param-collection factory)
        [loss activations] (logistic-regression m num-classes num-feats)
        get-data (data-generator num-classes num-feats)
        data (doall (take num-examples (repeatedly get-data)))
        test-data (doall (take num-examples (repeatedly get-data)))
        batch-gen #(partition num-batch data)
        predict-fn (module/predict-fn factory (module/static factory activations))
        loss-node-fn (fn [example]
                       (compute/with-inputs! factory loss example)
                       loss)
        train-opts {:num-iters num-iters
                    :learning-rate 0.1
                    :iter-reporter
                    [(report/accuracy
                      :test-accuracy
                      #(for [t test-data]
                         [(dissoc t "label") (first (get t "label"))])
                      predict-fn)]}]
    (train/train! m loss-node-fn batch-gen train-opts)))

(def cli-options
  ;; An option with a required argument
  [["-b" "--num-batch NUM" "Number of batches"
    :default 32
    :parse-fn #(Integer/parseInt %)]
   ["-n" "--num-examples NUM" "Number of elements"
    :default 1000
    :parse-fn #(Integer/parseInt %)]
   ["-i" "--num-iters NUM" "Number of iters"
    :default 10
    :parse-fn #(Integer/parseInt %)]
   ["-f" "--num-feats NUM" "Number of feats"
   :default 100
    :parse-fn #(Integer/parseInt %)]
   ["-h" "--help"]])

(comment
  (def opts {:num-examples 10000
             :num-batch 32
             :num-feats 10
             :num-iters 10}))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (println "Parse options: " (:options parse))
    (dotimes [i 10]
      (let [start (System/currentTimeMillis)]
        (println "Training " i)
        (train (:options parse))
        (let [time (- (System/currentTimeMillis) start)]
          (println "Took " time " msecs")
          (.flush System/out))))))
