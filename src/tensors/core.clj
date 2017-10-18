(ns tensors.core
  (:require [schema.core :as s])
  (:import [java.util Random]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tensor Shape

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

(s/defn effective-dimension [shape :- Shape]
  (let [n (count shape)
        num-squeeze-dims (count (filter #(= % 1) shape))]
    (- n num-squeeze-dims)))

(defn vector-shape? [shape]
  (= (count shape) 1))

(defn scalar-shape? [shape]
  (= shape [1]))

(defn validate-shape! [key expected-shape given-shape]
  (when-not (= expected-shape given-shape)
    (let [data {:expected expected-shape :given given-shape :key key}]
      (throw (ex-info "Got unexpected shape" data)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tensor Factory

(defprotocol PFactory
  (from-nums [this nums])
  (get-op [this op-key])
  (zeros [this shape])
  (transform!
    [this tensor get-val]
    [this tensor other-tensor get-val]
    "`get-val` can be a few different things, depending on arity

      Three argument version [this tensor get-val]
      ==============================
       * A fixed double to fill `tensor`
       * A IFn$ODD primitive function taking (dims, existing) which
         returns new value for the position


      Four argument version [this tensor other-tensor get-val]
      ================================
      Assumes other-tensor shape matches tensor
      * IFn$DDD takes (cur-val, other-val) and returns new value
      * IFn$ODDD takes (position, cur-val, other-val) and position
        is the long-array of the location")
  (->clj [this tensor])
  (grad-step! [this weight alpha grad])
  (copy-from-input! [this tensor! nums])
  (shape [this t]))
