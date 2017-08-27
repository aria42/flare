(ns tensors.core
  (:require [schema.core :as s])
  (:import [java.util Random]))

(s/defschema Shape
  [s/Int])

(defn guess-shape [nums]
  (cond
    (every? number? nums) [(count nums)]
    (every? coll? nums)
    (let [children-shapes (map guess-shape nums)
          [f & the-rest] children-shapes]
      (when-not (every? #(= f %) the-rest)
        (throw (RuntimeException. (str "Non-uniform shape: " nums))))
      (concat [(count nums)] f))
    :else nil))

(defprotocol PFactory
  (from-nums [this nums])
  (get-op [this op-key])
  (zeros [this shape]))

