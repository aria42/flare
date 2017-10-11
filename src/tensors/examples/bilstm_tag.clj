(ns tensors.examples.bilstm-tag
  (:require [tensors.node :as node]
            [tensors.rnn :as rnn]
            [tensors.compute :as compute]
            [tensors.embeddings :as embeddings]
            [clojure.java.io :as io]
            [clojure.tools.cli :refer [parse-opts]]
            [tensors.neanderthal-ops :as no]
            [tensors.model :as model]
            [tensors.computation-graph :as cg]))

(defn build-graph [words embed rnn-cell])

(def cli-options
  ;; An option with a required argument
  [["-f" "--file PATH" "path to data"
    :default "data/small-glove.50d.txt"]
   ["-s" "--emb-size NUM" "size of embedding data"
    :default 50]
   ["-l" "--lstm-size NUM" "lstm size"]])

(defn gen-sentence [n]
  (take n (repeatedly #(first (shuffle #{"the" "dog" "barks"})))))

(defn load-embeddings [opts]
  (let [factory (no/->Factory)]
    (embeddings/fixed-embedding
     (no/->Factory)
     (:emb-size opts)
     (-> opts :file io/reader embeddings/read-text-embedding-pairs))))

(defn graph-builder [model word-emb lstm-size]
  (let [emb-size (embeddings/embedding-size word-emb)
        cell (node/with-scope "forward"
               (rnn/lstm-cell model emb-size lstm-size))
        rev-cell (node/with-scope "reverse"
                   (rnn/lstm-cell model emb-size lstm-size))
        factory (model/tensor-factory model)]
    (fn [sent tag]
      (let [inputs (embeddings/sent-nodes factory word-emb sent)
            [fwd-outputs _] (rnn/build-seq cell sent-nodes)
            [rev-outputs _] (rnn/build-seq rev-cell (reverse sent-nodes))
            concat-hidden (cg/concat 0 (last fwd-outputs) (last rev-outputs))]
        concat-hidden))))

(comment
  (do 
    (def opts {:file "data/small-glove.50d.txt" :lstm-size 10 :emb-size 50})
    (def factory (no/->Factory))
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection factory))
    (def cell (rnn/lstm-cell model (:emb-size opts) (:lstm-size opts)))
    (def sent-words (gen-sentence 5))
    (def sent-nodes (mapv (fn [w]
                            (node/constant factory (embeddings/lookup emb w)))
                          sent-words))
    (def hiddens (rnn/build-seq cell sent-nodes))
    (def gb (graph-builder model emb 10))
    ))

(defn run [opts]
  (let [factory (no/->Factory)
        emb (load-embeddings opts)
        model (model/simple-param-collection factory)
        cell (rnn/lstm-cell model (:emb-size opts) (:lstm-size opts))
        sent-words (gen-sentence 10)
        sent-nodes (mapv (fn [w]
                           (node/constant
                            factory
                            (embeddings/lookup emb w)))
                         sent-words)
        hiddens (rnn/build-seq cell sent-nodes)
        ]))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (run (:options parse))))
