(ns tensors.examples.logistic-regression
  (:gen-class)
  (:require [tensors.neanderthal-ops :as no]
            [tensors.nd4j-ops :as nd4j-ops]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [tensors.model :as model]
            [tensors.graph-ops :as go]
            [tensors.train :as train]
            [tensors.computation-graph :as cg]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [clojure.tools.cli :refer [parse-opts]]))

(defn generate-data [num-datums num-classes num-feats]
  (let [r (java.util.Random. 0)
        W (dge num-classes num-feats)
        b (dv num-classes)]
    (alter! W (fn ^double [^long idx1 ^long idx2 ^double w] (.nextDouble r)))
    (alter! b (fn ^double [^long idx ^double w] (.nextDouble r)))
    (for [_ (range num-datums)]
      (let [rand-feats (mapv (fn [_] (if (.nextBoolean r) 1.0 0.0)) (range num-feats))
            f (dv rand-feats)
            class-idx (.nextInt r (int num-classes))
            activations (axpy (mv W f) b)
            label (imax activations)]
        {"f" rand-feats "label" [label]}))))

(defn lr-loss [m num-classes num-feats]
  (let [feat-vec (cg/input "f" [num-feats])
        activations (go/affine m feat-vec num-classes)
        label (cg/input "label" [1])]
    (go/cross-entropy-loss activations label)))

(defn train [{:keys [engine, num-examples, num-iters, num-feats, num-batch] :as opts}]
  (println "options " opts)
  (let [num-classes 5
        factory (case engine
                  :nd4j (nd4j-ops/->Factory)
                  :neanderthal (no/->Factory))
        m (model/simple-param-collection factory)
        loss (lr-loss m num-classes num-feats)
        loss (compute/compile-graph loss factory m)
        data (doall (generate-data num-examples num-classes num-feats))
        batch-gen #(partition num-batch data)]
    (train/sgd! m loss batch-gen {:num-iters num-iters :learning-rate 0.01})))

(def cli-options
  ;; An option with a required argument
  [["-e" "--engine ENGINE" "Engine {nd4j, neanderthal}"
    :default :nd4j
    :parse-fn keyword
    :validate [#{:nd4j, :neanderthal} "Must be {nd4j, neanderthal}"]]
   ["-b" "--num-batch NUM" "Number of batches"
    :default 32
    :parse-fn #(Integer/parseInt %)]
   ["-n" "--num-examples NUM" "Number of elements"
    :default 1000
    :parse-fn #(Integer/parseInt %)]
   ["-i" "--num-iters NUM" "Number of iters"
    :default 100
    :parse-fn #(Integer/parseInt %)]
   ["-f" "--num-feats NUM" "Number of feats"
   :default 100
    :parse-fn #(Integer/parseInt %)]
   ["-h" "--help"]])

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
