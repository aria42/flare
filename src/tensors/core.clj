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

(s/defn effective-dimension [shape :- Shape]
  (let [n (count shape)
        num-squeeze-dims (count (filter #(= % 1) shape))]
    (- n num-squeeze-dims)))

(defn vector-shape? [shape]
  (= (count shape) 1))

(defn scalar-shape? [shape]
  (= shape [1]))

(defprotocol PFactory
  (from-nums [this nums])
  (get-op [this op-key])
  (zeros [this shape])
  (fill! [this tensor new-val]
    "`new-val` can be a few different things
       * A fixed double
       * A IFn$ODD primitive function taking (dims, existing) which
         is a long-array or the current value as a double and returns
         new value a long-array of the indices of the current value,
         then the value at that position, then the function returns
         a new value for that position")
  (->clj [this tensor])
  (grad-step! [this weight alpha grad])
  (copy-from-input! [this tensor! nums]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Operations built on factory

(defn grad-clip! [factory grad max-val]
  (let [max-val (Math/abs (double max-val))
        min-val (- max-val)]
    (fill! factory grad
           (fn clip ^double [^longs indices ^double x]
             (if (> x max-val)
               max-val
               (if (< x min-val)
                 min-val
                 x))))))
