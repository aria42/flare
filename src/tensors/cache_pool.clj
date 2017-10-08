(ns tensors.cache-pool
  (:import [java.util.concurrent LinkedBlockingQueue ConcurrentHashMap]
           [java.util.function Function])
  (:require [tensors.cache-pool :as cache-pool]))

(defprotocol -CachePool
  (get-obj [this k])
  (obj-count [this k])
  (return-obj [this k v]))

(deftype MakeQueueFn [^long max-num]
  Function
  (apply [this _] (LinkedBlockingQueue. max-num)))

(defn make [max-num get-fn]
  (let [cmh (ConcurrentHashMap.)
        queue-fn (MakeQueueFn. (long max-num))]
    (reify -CachePool
      (get-obj [this k]
        (let [^LinkedBlockingQueue q (.computeIfAbsent cmh k ^Function queue-fn)
              return-fn (fn [x] (.offer q x))]
          (if-let [v (.poll q)]
            [v return-fn]
            [(get-fn k) return-fn])))
      (obj-count [this k]
        (if-let [^LinkedBlockingQueue q (.get cmh k)]
          (.size q)
          0))
      (return-obj [this k v]
        (let [^LinkedBlockingQueue q (.computeIfAbsent cmh k queue-fn)]
          (.offer q v))))))
