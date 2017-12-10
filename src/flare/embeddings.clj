(ns flare.embeddings
  (:require [flare.computation-graph :as cg]
            [flare.core :as flare]
            [flare.node :as node]))

(defprotocol Embedding
  (lookup [this obj])
  (vocab [this])
  (embedding-size [this]))

(defn sent-nodes
  "take a sentence and return sequence of constant node
   tensors, where each consant has the original word
   as part of the name.

   Will use `unk` if given for unknown tokens, or omit
   if `unk` isn't passed in"
  ([emb sent] (sent-nodes emb sent nil))
  ([emb sent unk]
   (for [word sent
         :let [e (lookup emb word)]
         :when (or e unk)]
     (node/const (node/gen-name "word") (or e unk)))))

(deftype FixedEmbedding [^java.util.Map m ^long emb-size]
  Embedding
  (lookup [this obj] (.get m obj))
  (vocab [this] (seq (.keySet m)))
  (embedding-size [this] emb-size)

  clojure.lang.Seqable
  (seq [this] (map (juxt key val) m)))

(defn fixed-embedding
  ([emb-size obj-vec-pairs]
   (fixed-embedding (:factory (flare/state)) emb-size obj-vec-pairs))
  ([factory emb-size obj-vec-pairs]
   (let [m (java.util.HashMap.)
         expected-shape [emb-size]]
     (doseq [[obj nums] obj-vec-pairs]
       (when-let [dupe (.get m obj)]
         (throw (ex-info "Duplicate entry" {:dupe obj})))
       (let [t (flare/from factory nums)
             s (flare/shape t)]
         (when (not= s expected-shape)
           (throw (ex-info "embedding doesn't have same shape"
                           {:expected [embedding-size] :actual s})))
         (.put m obj t)))
     (FixedEmbedding. m (long emb-size)))))

(defn learned-embedings
  [param-node vocab]
  (let [item->idx (into {} (map-indexed (fn [i x] [x i]) vocab))
        [num-items emb-size :as shape] (:shape param-node)
        param-val (:value param-node)
        param-grad (:grad param-node)
        param-emb-shape [emb-size]
        param-name (:ref-name param-node)]
    (when-not (and (= (count shape) 2) (= (count vocab) num-items))
      (throw (ex-info "Mismatched parameter embedding"
                      {:param-shape shape :vocab-size (count vocab)})))
    (reify Embedding
      (vocab [this] vocab)
      (embedding-size [this] emb-size)
      (lookup [this item]
        (when-let [idx (item->idx item)]
          (node/map->Node
           {:type :params
            :ref-name (format "%s@%s" param-name item)
            :shape param-emb-shape
            :value (flare/select! param-val [idx])
            :grad (flare/select! param-grad [idx])}))))))

(defn read-text-embedding-pairs [rdr]
  (for [^String line (line-seq rdr)
        :let [fields (.split line " ")]]
    [(aget fields 0)
     (map #(Double/parseDouble ^String %) (rest fields))]))
