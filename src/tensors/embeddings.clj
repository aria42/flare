(ns tensors.embeddings
  (:require [schema.core :as s]
            [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.node :as node]))

(defprotocol Embedding
  (lookup [this obj])
  (vocab [this])
  (embedding-size [this]))

(defn sent-nodes [factory emb sent]
  (for [word sent
        :let [e (lookup emb word)]
        :when e]
    (node/constant (node/gen-name "word") factory e)))

(deftype FixedEmbedding [^java.util.Map m ^long emb-size]
  Embedding
  (lookup [this obj] (.get m obj))
  (vocab [this] (seq (.keySet m)))
  (embedding-size [this] emb-size)

  clojure.lang.Seqable
  (seq [this] (map (juxt key val) m)))

(defn fixed-embedding
  [factory emb-size obj-vec-pairs]
  (let [m (java.util.HashMap.)
        expected-shape [emb-size]]
    (doseq [[obj nums] obj-vec-pairs]
      (when-let [dupe (.get m obj)]
        (throw (ex-info "Duplicate entry" {:dupe obj})))
      (let [t (tensors/from-nums factory nums)
            s (tensors/shape factory t)]
        (when (not= s expected-shape)
          (throw (ex-info "embedding doesn't have same shape"
                          {:expected [embedding-size] :actual s})))
        (.put m obj t)))
    (FixedEmbedding. m (long emb-size))))

(defn read-text-embedding-pairs [rdr]
  (for [^String line (line-seq rdr)
        :let [fields (.split line " ")]]
    [(aget fields 0)
     (map #(Double/parseDouble ^String %) (rest fields))]))
