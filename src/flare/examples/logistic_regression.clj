(set! *unchecked-math* true)
(ns flare.examples.logistic-regression
  (:gen-class)
  (:require [flare.neanderthal-ops :as no]
            [flare.compute :as compute]
            [flare.core :as flare]
            [flare.model :as model]
            [flare.module :as module]
            [flare.train :as train]
            [flare.computation-graph :as cg]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.tools.cli :refer [parse-opts]]
            [flare.node :as node]
            [flare.report :as report]))

(flare/init!)

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
        [f label]))))

(defn logistic-regression [model ^long num-classes ^long num-feats]
  (let
    [feats->activations (module/affine model num-classes [num-feats])]
    ;; A module knows how to make a graph from inputs
    (reify module/PModule
      ;; make class activations from feat-vec
      (graph [this feat-vec]
        ;; take feat-vec, make constant node, pass to affine
        (->> feat-vec
             (node/const "f")
             (module/graph feats->activations)))
      ;; make loss given feat-vec and label
      (graph [this feat-vec label]
        ;; cross-entropy between activations and label 
        (let [label-node (node/const "label" [label])
              activations (module/graph this feat-vec)]
          (cg/cross-entropy-loss activations label-node))))))

(defn train [{:keys [num-examples, num-classes, num-iters,
                     num-feats, num-batch]
              :as opts}]
  (println "options " opts)
  (let [m (model/simple-param-collection)
        classifier (logistic-regression m num-classes num-feats)
        get-data (data-generator num-classes num-feats)
        data (doall (take num-examples (repeatedly get-data)))
        test-data (doall (take num-examples (repeatedly get-data)))
        batch-gen #(partition num-batch data)
        predict-fn (module/predict-fn classifier)
        loss-node-fn (fn [[f label]]
                       (module/graph classifier f label))
        train-opts {:num-iters num-iters
                    :iter-reporters
                    [(report/accuracy :test
                                      (constantly test-data)
                                      predict-fn)]}]
    ;; write model to bytes, read from bytes
    ;; end-to-end test for model serialization
    (train/train! m loss-node-fn batch-gen train-opts)
    (let [baos (java.io.ByteArrayOutputStream.)
          _ (model/to-data! m baos)
          is (java.io.ByteArrayInputStream. (.toByteArray baos))
          m (model/simple-param-collection)
          classifier (logistic-regression m num-classes num-feats)]
      (model/from-data! m is)
      (println
       (report/gen
        (report/accuracy :test-reread
                         (constantly test-data)
                         predict-fn))))))

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

(def opts {:num-examples 10000
           :num-batch 32
           :num-classes 5
           :num-feats 10
           :num-iters 10})

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
