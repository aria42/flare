(ns flare.cache-pool)

(defprotocol CachePool
  (get-obj [this k])
  (obj-count [this k])
  (return-obj [this k v]))
