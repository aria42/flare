(ns tensors.graph
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [clojure.string :as str]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Walks

(defn bottom-up-walk [node walk-fn]
  (walk-fn node (map #(bottom-up-walk % walk-fn) (:children node))))

(defn top-down-walk [node walk-fn]
  ;; walk-fn can update children so do a let-binding
  (let [node (walk-fn node (:children node))]
    (assoc node
           :children (map #(top-down-walk % walk-fn) (:children node)))))

(defn post-order-nodes [target]
  (conj (vec (mapcat post-order-nodes (:children target))) target))

