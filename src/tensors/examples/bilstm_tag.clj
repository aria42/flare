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
            [tensors.computation-graph :as cg]
            [tensors.train :as train]))

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

(defn gen-sentence [n]
  (take n (repeatedly #(first (shuffle #{"the" "dog" "barks"})))))

(defn load-embeddings [opts]
  (let [factory (no/->Factory)]
    (embeddings/fixed-embedding
     (no/->Factory)
     (:emb-size opts)
     (-> opts :embed-file io/reader embeddings/read-text-embedding-pairs))))

(defn graph-builder [model word-emb lstm-size num-classes]
  (let [emb-size (embeddings/embedding-size word-emb)
        cell (node/with-scope "forward"
               (rnn/lstm-cell model emb-size lstm-size))
        rev-cell (node/with-scope "reverse"
                   (rnn/lstm-cell model emb-size lstm-size))
        factory (model/tensor-factory model)
        hidden-size (* 2 lstm-size)
        hidden->logits (model/add-params! model [num-classes hidden-size]
                                          :name  "hidden->logits")]
    (fn [[sent tag]]
      (let [inputs (embeddings/sent-nodes factory word-emb sent)
            [fwd-outputs _] (rnn/build-seq cell inputs)
            [rev-outputs _] (rnn/build-seq rev-cell (reverse inputs))
            concat-hidden (cg/concat 0 (last fwd-outputs) (last rev-outputs))
            logits (cg/* hidden->logits concat-hidden)]
        (when (seq inputs)
            (cg/cross-entropy-loss
             logits
             (node/constant "tag" factory [tag])))))))

(defn load-data [path]
  (for [line (line-seq (io/reader path))
        :let [[tag & sent] (.split (.trim ^String line) " ")]]
    [sent (Integer/parseInt tag)]))

(defn train [opts]
  (let [all-data (take (:num-data opts) (load-data (:train-file opts)))
        emb (load-embeddings opts)
        gen-batches #(partition 32 all-data)
        factory (no/->Factory)
        m (model/simple-param-collection factory)
        ;; need to provide forward-computed graph for loss
        gb (comp
            (fn [n] (compute/forward-pass! n m))
            (graph-builder m emb (:lstm-size opts) (:num-classes opts)))
        train-opts {:num-iters 100 :learning-rate 0.01}]
    (train/sgd! m gb gen-batches train-opts)))

(comment
  (do
    (def opts {:embed-file "data/small-glove.50d.txt"
               :lstm-size 25
               :num-classes 2
               :train-file "data/sentiment-10k.txt"
               :emb-size 50})
    (def factory (no/->Factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def cell (rnn/lstm-cell model (:emb-size opts) (:lstm-size opts)))
    (def sent-words (gen-sentence 5))
    (def sent-nodes (mapv (fn [w]
                            (node/constant factory (embeddings/lookup emb w)))
                          sent-words))
    (def hiddens (rnn/build-seq cell sent-nodes))
    (def gb (graph-builder model emb (:lstm-size opts) (:num-classes opts)))
    ))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (train (:options parse))))
