(ns tensors.examples.bilstm-tag
  (:gen-class)
  (:require [tensors.node :as node]
            [tensors.rnn :as rnn]
            [tensors.compute :as compute]
            [tensors.embeddings :as embeddings]
            [clojure.java.io :as io]
            [clojure.tools.cli :refer [parse-opts]]
            [tensors.neanderthal-ops :as no]
            [tensors.model :as model]
            [tensors.report :as report]
            [tensors.computation-graph :as cg]
            [tensors.train :as train]
            [tensors.module :as module]))

(defn build-graph [words embed rnn-cell])

(def cli-options
  ;; An option with a required argument
  [["-d" "--train-file PATH" "path to data"
    :default "data/sentiment-10k.txt"]
   ["-e" "--embed-file PATH" "path to data"
    :default "data/small-glove.50d.txt"]
   ["-n" "--num-classes PATH" "path to data"
    :default 2
    :parse-fn #(Integer/parseInt ^String %)]
   ["-s" "--emb-size NUM" "size of embedding data"
    :default 50
    :parse-fn #(Integer/parseInt ^String %)]
   ["-l" "--lstm-size NUM" "lstm size"
    :default 25
    :parse-fn #(Integer/parseInt ^String %)]
   ["-x" "--num-data DATA"
    :default 10000
    :parse-fn #(Integer/parseInt ^String %)]])

(defn load-embeddings [opts]
  (let [factory (no/factory)]
    (embeddings/fixed-embedding
     (no/factory)
     (:emb-size opts)
     (-> opts :embed-file io/reader embeddings/read-text-embedding-pairs))))

(defn stupid-lstm-sent-classifier [model word-emb lstm-size num-classes]
  (let [emb-size (embeddings/embedding-size word-emb)
        cell nil #_(node/with-scope "simple-forward"
                 (rnn/simple-lstm-cell model emb-size lstm-size))
        ;; rev-cell (node/with-scope "reverse"
        ;;           (rnn/lstm-cell model emb-size lstm-size))
        factory (model/tensor-factory model)
        num-dirs 1
        hidden-size (* num-dirs lstm-size)
        hidden->logits (node/with-scope "hidden->logits"
                         (module/affine model num-classes [emb-size]))]
    (reify
      module/Module
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes factory word-emb sent))]
          (let [
                ;;[fwd-outputs _] (rnn/build-seq cell inputs)
                ;; [rev-outputs _] (rnn/build-seq rev-cell (reverse inputs))
                ;; concat-hidden (cg/concat 0 (last fwd-outputs) (last rev-outputs))
                ;; hidden (last fwd-outputs)
                ]
            (module/graph hidden->logits (last inputs))
            #_(module/graph hidden->logits hidden))))
      ;; build loss node for two-arguments
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/constant "label" factory [label])]
            (cg/cross-entropy-loss logits label-node)))))))

(defn lstm-sent-classifier [model word-emb lstm-size num-classes]
  (let [emb-size (embeddings/embedding-size word-emb)
        cell (node/with-scope "simple-forward"
                 (rnn/lstm-cell model emb-size lstm-size))
        rev-cell (node/with-scope "reverse"
                   (rnn/lstm-cell model emb-size lstm-size))
        factory (model/tensor-factory model)
        num-dirs 2
        hidden-size (* num-dirs lstm-size)
        hidden->logits (node/with-scope "hidden->logits"
                         (module/affine model num-classes [hidden-size]))]
    (reify
      module/Module
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes factory word-emb sent))]
          (let [
                [fwd-outputs _] (rnn/build-seq cell inputs)
                [rev-outputs _] (rnn/build-seq rev-cell (reverse inputs))
                hidden (cg/concat 0 (last fwd-outputs) (last rev-outputs))
                ]
            #_(module/graph hidden->logits (last inputs))
            (module/graph hidden->logits hidden))))
      ;; build loss node for two-arguments
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/constant "label" factory [label])]
            (cg/cross-entropy-loss logits label-node)))))))

(defn load-data [path]
  (for [line (line-seq (io/reader path))
        :let [[tag & sent] (.split (.trim ^String line) " ")]]
    [(take 5 sent) (double (Integer/parseInt tag))]))

(defn train [opts]
  (let [emb (load-embeddings opts)
        train-data (take (:num-data opts) (load-data (:train-file opts)))
        test-data (take (:num-data opts) (load-data (:test-file opts)))
        gen-batches #(partition-all 1000 train-data)
        factory (no/factory)
        m (model/simple-param-collection factory)
        ;; need to provide forward-computed graph for loss
        classifier (lstm-sent-classifier m emb (:lstm-size opts) (:num-classes opts))
        gb (fn [[sent tag]]
             (module/forward! factory classifier sent tag))
        train-opts {:num-iters 100
                    :iter-reporter (report/test-accuracy
                                    (constantly train-data)
                                    (fn [sent]
                                      (module/predict factory classifier sent)))
                    :learning-rate 1}]
    (println "Params " (map first (seq m)))
    (println "Total " (model/total-num-params m))
    (train/train! m gb gen-batches train-opts)))

(defn build-test [lstm-classifier-fn]
  (do
    (def opts {:embed-file "data/small-glove.50d.txt"
               :lstm-size 10
               :num-classes 2
               :num-data 100
               :train-file "data/sentiment-train10k.txt"
               :test-file "data/sentiment-test10k.txt"
               :emb-size 50})
    (def factory (no/factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def train-data (take (:num-data opts) (load-data (:train-file opts))))
    (def classifier (lstm-classifier-fn model emb (:lstm-size opts) (:num-classes opts)))
    (def gb (fn [[sent tag]]
              (module/graph classifier sent tag)))
    (require '[tensors.optimize :as optimzie])
    (def lf (optimize/loss-fn model gb (take 1 train-data)))
    (def xs (double-array (model/total-num-params model)))
    {:gb gb :lf lf :xs xs :m model}
    ))

(defn do-bump-test []
  (do
    (def opts {:embed-file "data/small-glove.50d.txt"
               :lstm-size 10
               :num-classes 2
               :num-data 100
               :train-file "data/sentiment-train10k.txt"
               :test-file "data/sentiment-test10k.txt"
               :emb-size 50})
    (def factory (no/factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def train-data (take (:num-data opts) (load-data (:train-file opts))))
    (def classifier (lstm-sent-classifier model emb (:lstm-size opts) (:num-classes opts)))
    (def gb (fn [[sent tag]]
              (module/graph classifier sent tag)))
    (require '[tensors.optimize :as optimzie])
    (def lf (optimize/loss-fn model gb (take 1 train-data)))
    (def xs (double-array (repeat (model/total-num-params model) 0.0))
      #_(model/to-doubles model))
    (dotimes [_ 10]
      (optimize/bump-test lf xs 0))
    ))

(comment
  (do
    (def opts {:embed-file "data/small-glove.50d.txt"
               :lstm-size 10
               :num-classes 2
               :num-data 100
               :train-file "data/sentiment-train10k.txt"
               :test-file "data/sentiment-test10k.txt"
               :emb-size 50})
    (def factory (no/factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def train-data (take (:num-data opts) (load-data (:train-file opts))))
    (def classifier (lstm-sent-classifier model emb (:lstm-size opts) (:num-classes opts)))
    (def gb (fn [[sent tag]]
              (module/graph classifier sent tag)))
    (require '[tensors.optimize :as optimzie])
    (def lf (optimize/loss-fn model gb (take 1 train-data)))
    (def xs (model/to-doubles model))
    #_(optimize/bump-test lf xs 0)
    ))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (train (:options parse))))
