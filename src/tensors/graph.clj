(ns tensors.graph  
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [clojure.string :as str]
            [tensors.graph :as graph]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Walks

(defn bottom-up-walk [node walk-fn]
  (walk-fn node (map #(bottom-up-walk % walk-fn) (:children node))))

(defn post-order-nodes [target]
  (conj (vec (mapcat post-order-nodes (:children target))) target))

