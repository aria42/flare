(ns tensors.core
  (:require [schema.core :as s]
            [tensors.cache-pool :as cache-pool])
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
  "A `PFactory` knows how to make and manipulate tensors. The tensor
   type depends on the `PFactory`, but there are some conventions
   tensors should have:

      * Tensors should be seqable (might revisit this)
      * Tensors are mutable objects (duh, performance) "
  (from [this data]
    "create a tensor from `data`. Should minimally accept
     (nested) sequences of numbers, but can also effectively
     copy an existing tensor")
  (get-op [this op-key]
    "returns the `TensorOp` associated with the `op-key`")
  (zeros [this shape]
    "create a 0.0 filled tensor of a given shape")
  (transform!
    [this tensor get-val]
    [this tensor other-tensor get-val]
    "In-place transform of a `tensor` using the `get-val` function,
     which can be a few different things

      [this tensor get-val]
      ==============================
       * A fixed double to fill `tensor`
       * A `IFn$ODD` primitive function taking (dims, existing) which
         returns new value for the position


      [this tensor other-tensor get-val]
      ================================
      Assumes other-tensor shape matches tensor
      * `IFn$DDD` takes (cur-val, other-val) and returns new value
      * `IFn$ODDD` takes (position, cur-val, other-val) and position
        is the long-array of the location")
  (copy! [this dst-tensor! src-tensor-like]
    "copy from source to destination. The src should be same set of things
     accdepted by `PFactory/from`")
  (shape [this t]
    "return the shape of the tennsor as integer (clojure) vector"))

(def *factory*
  "atom for global factory to avoid passing into operations"
  (atom nil))


(defn set-factory! [factory]
  (reset! *factory* factory))

(defn with-cache [factory num-to-cache]
  (with-meta factory
    {:cache (cache-pool/make
             (or num-to-cache 100)
             (fn [shape] (zeros factory shape)))}))
