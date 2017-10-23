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
    :default "data/sentiment-train10k.txt"]
   ["-sss" "--test-file PATH" "path to test"
    :default "data/sentiment-test10k.txt"]
   ["-e" "--embed-file PATH" "path to data"
    :default "data/small-glove.100d.txt"]
   ["-n" "--num-classes PATH" "path to data"
    :default 2
    :parse-fn #(Integer/parseInt ^String %)]
   ["-s" "--emb-size NUM" "size of embedding data"
    :default 100
    :parse-fn #(Integer/parseInt ^String %)]
   ["-l" "--lstm-size NUM" "lstm size"
    :default 25
    :parse-fn #(Integer/parseInt ^String %)]
   ["-x" "--num-data DATA"
    :default 2000
    :parse-fn #(Integer/parseInt ^String %)]])

(defn load-embeddings [opts]
  (let [factory (no/factory)]
    (embeddings/fixed-embedding
     (no/factory)
     (:emb-size opts)
     (-> opts :embed-file io/reader embeddings/read-text-embedding-pairs))))

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
          (let [[fwd-outputs _] (rnn/build-seq cell inputs)
                [rev-outputs _] (rnn/build-seq rev-cell (reverse inputs))
                train? (:train? (meta this))
                hidden (cg/concat 0 (last fwd-outputs) (last rev-outputs))
                hidden (if train? (cg/dropout 0.5 hidden) hidden)]
            (module/graph hidden->logits hidden))))
      ;; build loss node for two-arguments
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/constant "label" factory [label])]
            (cg/cross-entropy-loss logits label-node)))))))

(defn load-data [path]
  (for [line (line-seq (io/reader path))
        :let [[tag & sent] (.split (.trim ^String line) " ")]]
    [sent (double (Integer/parseInt tag))]))

(defn train [opts]
  (let [emb (load-embeddings opts)
        train-data (take (:num-data opts) (load-data (:train-file opts)))
        test-data (take (:num-data opts) (load-data (:test-file opts)))
        gen-batches #(partition-all 32 train-data)
        factory (no/factory)
        m (model/simple-param-collection factory)
        ;; need to provide forward-computed graph for loss
        classifier (lstm-sent-classifier m emb (:lstm-size opts) (:num-classes opts))
        gb (fn [[sent tag]]
             (-> classifier
                 (with-meta {:train? true})
                 (module/graph sent tag )))
        train-opts {:num-iters 100
                    :iter-reporter (report/concat
                                    (report/test-accuracy :train-accuracy
                                                          (constantly train-data)
                                                          (fn [sent]
                                                            (module/predict factory classifier sent)))
                                    (report/test-accuracy :test-accuracy
                                                          (constantly test-data)
                                                          (fn [sent]
                                                            (module/predict factory classifier sent))))
                    :learning-rate 1}]
    (println "Params " (map first (seq m)))
    (println "Total " (model/total-num-params m))
    (train/train! m gb gen-batches train-opts)))

(comment
  (do
    (def opts {:embed-file "data/small-glove.100d.txt"
               :lstm-size 25
               :num-classes 2
               :num-data 2000
               :train-file "data/sentiment-train10k.txt"
               :test-file "data/sentiment-test10k.txt"
               :emb-size 100})
    (def factory (no/factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def train-data (take (:num-data opts) (load-data (:train-file opts))))
    (def classifier (lstm-sent-classifier model emb (:lstm-size opts) (:num-classes opts)))
    (def gb (fn [[sent tag]]
              (module/graph classifier sent tag)))
    (require '[tensors.optimize :as optimzie])
    #_(def lf (optimize/loss-fn model gb (take 1 train-data)))
    #_(def xs (model/to-doubles model))
    #_(optimize/bump-test lf xs 0)
    ))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (println (:options parse))
    (train (:options parse))))
