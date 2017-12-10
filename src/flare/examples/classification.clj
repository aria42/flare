(ns flare.examples.classification
  (:gen-class)
  (:require [flare.node :as node]
            [flare.cnn :as cnn]
            [flare.rnn :as rnn]
            [flare.compute :as compute]
            [flare.embeddings :as embeddings]
            [clojure.java.io :as io]
            [clojure.tools.cli :refer [parse-opts]]
            [flare.neanderthal-ops :as no]
            [flare.model :as model]
            [flare.report :as report]
            [flare.computation-graph :as cg]
            [flare.train :as train]
            [flare.module :as module]
            [flare.optimize :as optimize]
            [flare.core :as flare]))

(def cli-options
  ;; An option with a required argument
  [["-train" "--train-file PATH" "path to data"
    :default "data/sentiment-train10k.txt"]
   ["-test" "--test-file PATH" "path to test"
    :default "data/sentiment-test10k.txt"]
   ["-e" "--embed-file PATH" "path to data"
    :default "data/small-glove.300d.txt"]
   ["-c" "--num-classes PATH" "path to data"
    :default 2
    :parse-fn #(Integer/parseInt ^String %)]
   ["s" "--emb-size NUM" "size of embedding data"
    :default 300
    :parse-fn #(Integer/parseInt ^String %)]
   ["m" "--model-type MODEL_TYPE" "bilstm or cnn"
    :default :bilstm
    :parse-fn keyword]
   ["l" "--lstm-size NUM" "lstm size"
    :default 25
    :parse-fn #(Integer/parseInt ^String %)]
   ["-n"  "--num-data DATA"
    :default 2000
    :parse-fn #(Integer/parseInt ^String %)]])

(defn load-embeddings [opts]
  (embeddings/fixed-embedding
   (no/factory)
   (:emb-size opts)
   (-> opts :embed-file io/reader embeddings/read-text-embedding-pairs)))

(flare/init!)

(defn lstm-sent-classifier 
  [model word-emb ^long lstm-size ^long num-classes]
  (node/let-scope
      ;; let-scope so the parameters get smart-nesting
      [^long emb-size (embeddings/embedding-size word-emb)
       ^long num-dirs 2
       input-size (* num-dirs emb-size)
       hidden-size (* num-dirs lstm-size)
       lstm (rnn/lstm-cell model input-size hidden-size)
       hidden->logits (module/affine model num-classes [hidden-size])]
    (reify
      module/PModule
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes word-emb sent))]
          (let [hiddens (rnn/build-seq lstm inputs (= num-dirs 2))
                train? (:train? (meta this))
                ;; take last output as hidden
                hidden (last (map first hiddens))
                hidden (if train? (cg/dropout 0.5 hidden) hidden)]
            (module/graph hidden->logits hidden))))
      ;; build loss node for two-arguments
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/const "label" [label])]
            (cg/cross-entropy-loss logits label-node)))))))

(defn cnn-sent-classifier 
  [model word-emb ^long num-classes]
  (node/let-scope
      ;; let-scope so the parameters get smart-nesting
      [^long emb-size (embeddings/embedding-size word-emb)
       cnn-feats (cnn/cnn-1D-feats model emb-size
                                   [{:width 1 :height 25}
                                    {:width 2 :height 25}
                                    {:width 3 :height 25}])
       ;; 75 is number of concattened cnn feats
       hidden->logits (module/affine model num-classes [75])]
    (reify
      module/PModule
      ;; build logits
      (graph [this sent]
        (when-let [inputs (seq (embeddings/sent-nodes word-emb sent))]
          (let [hidden (module/graph cnn-feats inputs)
                train? (:train? (meta this))
                hidden (if train? (cg/dropout 0.5 hidden) hidden)]
            (module/graph hidden->logits hidden))))
      ;; build loss node for two-arguments
      (graph [this sent label]
        (when-let [logits (module/graph this sent)]
          (let [label-node (node/const "label" [label])]
            (cg/cross-entropy-loss logits label-node)))))))

(defn load-data [path]
  (for [^String line (line-seq (io/reader path))
        :let [[tag & sent] (.split (.trim line) " ")]]
    [sent (double (Integer/parseInt tag))]))

(defn get-classifier [model-type model emb lstm-size num-classes]
  (case model-type
    :bilstm (lstm-sent-classifier model emb lstm-size num-classes)
    :cnn (cnn-sent-classifier model emb num-classes)))

(defn train [{:keys [lstm-size, num-classes, train-file, test-file model-type] 
              :as opts}]
  (let [emb (load-embeddings opts)
        train-data (take (:num-data opts) (load-data train-file))
        test-data (take (:num-data opts) (load-data test-file))
        gen-batches #(partition-all 32 train-data)
        model (model/simple-param-collection)
        ;; classifier can use a cache to avoid
        ;; re-allocating tensors across prediction
        classifier (get-classifier model-type model emb lstm-size num-classes)
        loss-fn (fn [[sent tag]]
                    (-> classifier
                        (with-meta {:train? true})
                        (module/graph sent tag)))
        predict-fn (module/predict-fn classifier)
        train-opts
          {:num-iters 100
           ;; report train/test accuracy each iter
           :iter-reporters
           [;; Train Accuracy
            (report/accuracy :train (constantly train-data) predict-fn)
            ;; Test Accuracy
            (report/accuracy :test (constantly test-data) predict-fn)
            ;; Report performance info on tensor-ops
            (report/callback #(-> (flare/state) :factory flare/debug-info))]
           :learning-rate 1}]
    ;; prints shape of all parameters 
    (println "Params " (map (juxt first (comp :shape second)) (seq model)))
    (println "Total # params " (model/total-num-params model))
    (train/train! model loss-fn gen-batches train-opts)))

(defn -main [& args]
  (let [parse (parse-opts args cli-options)]
    (println (:options parse))
    (train (:options parse))))

(comment

  (def opts {:embed-file "data/small-glove.50d.txt"
             :lstm-size 100
             :num-classes 2
             :model-type :cnn
             :num-data 1000
             :train-file "data/sentiment-train10k.txt"
             :test-file "data/sentiment-test10k.txt"
             :emb-size 50})
  ;; Hack to test LSTM end-to-end gradient
  (do
    (def emb (load-embeddings opts))
    (def model (model/simple-param-collection))
    (def classifier (lstm-sent-lassifier model emb 10 2))
    (def loss-fn (fn [[sent tag]]
                   (-> classifier
                       (with-meta {:train? true})
                       (module/graph sent tag))))
    (def df (optimize/loss-fn model loss-fn train-data))
    (optimize/rand-bump-test df (model/to-doubles model))))
