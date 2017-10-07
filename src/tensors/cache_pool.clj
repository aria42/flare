(ns tensors.cache-pool
  (:import [java.util.concurrent LinkedBlockingQueue ConcurrentHashMap]
           [java.util.function Function])
  (:require [tensors.cache-pool :as cache-pool]))

(defprotocol -CachePool
  (get-obj [this k])
  (obj-count [this k])
  (return-obj [this k v]))

(defn make [max-num get-fn]
  (let [cmh (ConcurrentHashMap.)
        queue-fn (reify Function
                   (apply [this _]
                     (LinkedBlockingQueue. (int max-num))))]
    (reify -CachePool
      (get-obj [this k]
        (let [^LinkedBlockingQueue q (.computeIfAbsent cmh k queue-fn)]
          (if-let [v (.poll q)]
            (do
              #_(println "Cache hit for " k " count now " (cache-pool/obj-count this k))
              v)
            (get-fn k))))
      (obj-count [this k]
        (if-let [^LinkedBlockingQueue q (.get cmh k)]
          (.size q)
          0))
      (return-obj [this k v]
        (let [^LinkedBlockingQueue q (.computeIfAbsent cmh k queue-fn)
              cnt (cache-pool/obj-count this k)]
          (.offer q v)
          #_(println "Return cache on key " k " count: " cnt " -> " (cache-pool/obj-count this k)))))))
