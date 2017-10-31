(ns flare.core
  (:require [schema.core :as s]
            [flare.cache-pool :as cache-pool])
  (:import [java.util Random LinkedList]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tensor Factory

(defprotocol PTensorFactory
  "A `PTensorFactory` knows how to make and manipulate tensors. The tensor
   type depends on the `PTensorFactory`, but there are some conventions
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
     accdepted by `PTensorFactory/from`")
  (shape [this t]
    "return the shape of the tennsor as integer (clojure) vector"))


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

(defn validate-shape!
  ([expected-shape given-shape]
   (validate-shape! :bad-shape expected-shape given-shape))
  ([key expected-shape given-shape]
   (when-not (= expected-shape given-shape)
     (let [data {:expected expected-shape :given given-shape :key key}]
       (throw (ex-info "Got unexpected shape" data))))))

(defprotocol -InternalPTensorFactory
  "protocols for internal understanding of performance/correctness,
   not meant to be used except by 'experts'"
  (debug-info [this] "map of debug/perf info"))

(def *factory
  "atom for global factory to avoid passing into operations"
  (atom nil))


(defn set-factory! [factory]
  (reset! *factory factory))

(defn cache [factory num-to-cache]
  (let [m (java.util.HashMap.)]
    (reify
      cache-pool/-CachePool
      (get-obj [this shape]
        (if-let [^LinkedList lst (.get m shape)]
          (if-let [t (.poll lst)]
            (do (transform! factory t 0.0)
                t)
            (zeros factory shape))
          (zeros factory shape)))
      (return-obj [this shape t]
        (if-let [^LinkedList lst (.get m shape)]
          (.offer lst t)
          (.put m shape (doto (LinkedList.) (.add t)))))
      (obj-count [this shape]
        (if-let [^LinkedList lst (.get m shape)]
          (.size lst)
          0)))))
