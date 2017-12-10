(ns flare.core
  (:refer-clojure :exclude [set!])
  (:require [clojure.spec.alpha :as s]
            [flare.cache-pool :as cache-pool])
  (:import [java.util LinkedList]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tensor and Factory protocols

(defprotocol PTensorFactory
  "A `PTensorFactory` knows how to make and manipulate tensors. The tensor
   type depends on the `PTensorFactory`, but there are some conventions
   tensors should have:

      * Tensors should be seqable (might revisit this)
      * Tensors are mutable objects (duh, performance)

  For external API use, use the `zeros` and `from` wrappers
  below which can default the tensor factory"
  (get-op [this op-key]
    "returns the `TensorOp` associated with the `op-key`")
  (-from [this data]
    "create a tensor from `data` (satisfies `Tensor`). Should minimally accept
     (nested) sequences of numbers, but can also effectively
     copy an existing tensor")
  (-zeros [this shape]
    "create a 0.0 filled tensor of a given shape"))

(defprotocol Tensor
  (factory [this]
    "returns `PTensorFactory` underlying this tensor")
  (shape [this]
    "return the shape of the tennsor as integer (clojure) vector")
  (add
    [this other]
    [this scale other]
    "return new tensor which adds this to `other`
     and possibly scales `other` by scalar `scale`")
  (add!
    [this other]
    [this scale other]
    "mutable version of `add` that updates `this`")
  (div
    [this denom]
    [this denom-offset denom]
    [this numer-offset other denom-offset]
    "return new tensor element-wise divide by `denom` possibly with
     offsets for numerator or denominator offset
    result = (this + numer-offset) / (other + denom-offset)")
  (div!
    [this denom]
    [this denom-offset denom]
    [this numer-offset other denom-offset]
    "mutable version of `div`")
  (mult
    [this other]
    "return new pointwise multiplication tensor")
  (mult!
    [this other]
    "pointwise in-place multiplication")
  (pow
    [this exp]
    "element-wise exponeniate")
  (pow!
    [this exp]
    "in-place element-wise exponetiationate")
  (scale [this alpha]
    "return sclaed tensor")
  (scale! [this alpha]
    "mutate scale")
  (transform
    [this get-val]
    [this other-tensor get-val]
    "Create a new `tensor` using the `get-val` function,
     which can be a few different things

      [this get-val]
      ==============================
       * A fixed double to fill `tensor`
       * A `IFn$ODD` primitive function taking (dims, existing) which
         returns new value for the position


      [this other-tensor get-val]
      ================================
      Assumes other-tensor shape matches tensor
      * `IFn$DDD` takes (cur-val, other-val) and returns new value
      * `IFn$ODDD` takes (position, cur-val, other-val) and position
        is the long-array of the location")
  (transform!
    [this get-val]
    [this other-tensor get-val]
    "mutable version of `transform!` that updates `this`")
  (copy! [this src-tensor-like]
    "copy from source to this tensor. The src should be same set of things
     accdepted by `PTensorFactory/from`"))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Tensor Shape

(s/def ::shape (s/coll-of int? :gen-max 4))

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

(defn cache [factory num-to-cache]
  (let [m (java.util.HashMap.)
        zero (Double. 0.0)]
    (reify
      cache-pool/CachePool
      (get-obj [this shape]
        (if-let [^LinkedList lst (.get m shape)]
          (if-let [t (.poll lst)]
            (do (transform! t zero)
                t)
            (-zeros factory shape))
          (-zeros factory shape)))
      (return-obj [this shape t]
        (if-let [^LinkedList lst (.get m shape)]
          (.offer lst t)
          (.put m shape (doto (LinkedList.) (.add t)))))
      (obj-count [this shape]
        (if-let [^LinkedList lst (.get m shape)]
          (.size lst)
          0)))))

(s/def ::state
  (s/keys :req-un [::factory, ::eager?, ::cache]))

(s/def ::factory (partial satisfies? PTensorFactory))
(s/def ::eager? #{true false})
(s/def ::cache (partial satisfies? cache-pool/CachePool))

(def ^:private *state
  "global state for flow"
  (atom nil))

(defn state []
  (let [state @*state]
    (when (nil? state)
      (throw (ex-info "Didn't call `flare/set!` or `flare/init!`" {})))
    (when (and (:eager? state) (nil? (:factory state)))
      (throw (ex-info "Can't be eager with setting :factory"
                      {:state state})))
    state))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Maain Functions

(defn set!
  [init-state]
  (reset! *state init-state))

(defn zeros
  "create a zero tensor with given shape using default tensor factory.
   Unless `:no-cache?` is passed in will use cache if avaialble"
  ([shape]
   (zeros (:factory (state)) shape))
  ([factory shape & {:keys [no-cache?]}]
   (if (or no-cache? (nil? (:cache (state))))
     (-zeros factory shape)
     (let [c (:cache (state))]
       (cache-pool/get-obj c shape)))))

(defn from
  "coerce data into a tensor, should work with nested clojure seqs of numbers"
  ([factory data]
   (-from factory data))
  ([data]
   (-from (:factory (state)) data)))

(defn init!
  "Defaults the state of flare with following
     * eager? is true, so all graph ops eagerly compute forward
     * factory is neanderthal (only good tensor impl)
     * a cache with `num-to-cache` (defaults to 1,000)"
  ([] (init! 1000))
  ([num-to-cache]
   (let [ns-symb (symbol 'flare.neanderthal-ops)
         _ (require ns-symb)
         ns (find-ns ns-symb)
         factory ((ns-resolve ns 'factory))]
     (reset! *state
             {:eager? true
              :factory factory
              :cache (cache factory num-to-cache)}))))
