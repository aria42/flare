(ns tensors.model
  (:import [java.util HashMap Map]
           [tensors.node Node])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.cache-pool :as cache-pool]
            [tensors.model :as model]
            [tensors.node :as node]
            [plumbing.core :as p])
  (:import [java.util Arrays]))

(s/defschema InitParamSpec
  "Spec for how to generate parameter entries independently
  implement `get-param-rng` multi-method for `:distribution`
  for a new distirbution"
  {:distribution s/Keyword
   s/Any s/Any})

(defmulti ^clojure.lang.IFn$ODD get-param-rng :distribution)

(defmethod get-param-rng :uniform
  [{:keys [rand-seed, lower, upper]}]
  (let [lower (double (or lower -1.0))
        upper (double (or upper 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double [^longs indices ^double x]
      (+ lower (* (- upper lower) (.nextDouble r))))))

(defmethod get-param-rng :normal
  [{:keys [rand-seed, mean, sigma]}]
  (let [mean (double (or mean 0.0))
        sigma (double (or sigma 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double [^longs indices ^double x]
      (/ (+ (.nextGaussian r) mean) sigma))))

(defprotocol PModel
  (tensor-factory [this]
    "return the underlying tensor factory for the model")
  (-add-params! [this param-name shape init-spec]
    "add parameters to the model, returns a param graph node. Some
     argument defaulting happens below so this is the internal method")
  (-add-param-metadata! [this param-name key val])
  (canonical-node [this param-name]
    "returns a caonical `Node` for the parameter. If parameters
     have been initialized, also returns `:value` and `:grad` tensor fields"))

(defn total-num-params [model]
  (p/sum (fn [[_ p]] (apply * (:shape p))) model))

(defprotocol -TestPModel
  "only for testing with models"
  (fix-param! [this param-name value]))

(defn add-params!
  [model shape & {:keys [name, init]}]
  (let [name (or name (clojure.core/name (gensym "param")))
        init (or init {:distribution :uniform})]
    (s/validate s/Str name)
    (s/validate InitParamSpec init)
    (-add-params! model name shape init)))

(defn with-metadata!
  [model param-node-or-name key value]
  (let [node-name (if (instance? Node param-node-or-name)
                    (.ref-name ^Node param-node-or-name)
                    param-node-or-name)]
    (-add-param-metadata! model node-name key value)))

(defn rand-params! [model]
  (doseq [[_ node] (seq model)]
    (when-let [init (:init node)]
      (tensors/transform!
       (tensor-factory model)
       (p/safe-get  node :value)
       (get-param-rng init)))))

(s/defn simple-param-collection :- PModel
  "Simple collection of parameters
   NOTE: The meta-data of the param-collection gives you access
   to the underlying data. Don't use it except for an emergency!

  The factory is also adorned with a caching pool under the
  :cache meta-data "
  [factory :- tensors/PFactory]
  (let [m (java.util.HashMap.)]
    (with-meta
      (reify

        PModel
        (tensor-factory [this] factory)
        (-add-params! [this param-name shape init-spec]
          (let [param-name (node/scoped-name param-name)]
            (when-let [existing (.get m param-name)]
              (throw (ex-info "Existing param key" {:existing existing})))
            (let [node (node/map->Node
                        {:type :params
                         :ref-name param-name
                         :value (tensors/zeros factory shape)
                         :grad (tensors/zeros factory shape)
                         :factory factory
                         :shape shape
                         :init init-spec})
                  get-param-val (get-param-rng init-spec)]
              ;; initialize param vals from init-spec
              (tensors/transform! factory (:value node) get-param-val)
              (.put m param-name node)
              node)))
        (-add-param-metadata! [this param-name key val]
          (let [n (canonical-node this param-name)
                n (with-meta n (assoc (meta n) key val))]
            (.put m param-name n)
            n))
        (canonical-node [this param-or-name] (.get m param-or-name))

        -TestPModel
        (fix-param! [this param-or-name tensor-like]
          (let [param-name (if (instance? Node param-or-name)
                             (.ref-name ^Node param-or-name)
                             param-or-name)]

            (when-let [param (.get m param-name)]
              (let [param-tensor (:value param)]
                (tensors/copy-from-input! factory param-tensor tensor-like)))))
        clojure.lang.Seqable
        (seq [this]
          (for [e m] [(key e) (val e)])))
      {:data m})))


(defn to-doubles
  "Flatten parameters into a single vector"
  [model]
  (let [factory (model/tensor-factory model)
        xs (double-array (total-num-params model))
        mapping (into {} (seq model))
        es (sort-by key mapping)]
    (loop [es es offset 0]
      (if-let [e (first es)]
        (let [[k n] e
              v (flatten (seq (:value n)))]
          (System/arraycopy
           (double-array v)
           0
           xs
           offset
           (count v))
          (recur (next es) (+ offset (count v))))
        xs))))

(defn from-doubles!
  "Flatten parameters into a single vector"
  [model ^doubles xs]
  (when-not (= (alength xs) (total-num-params model))
    (throw (ex-info "Array doesn't match model size"
                    {:model-size (total-num-params model)
                     :array-len (alength xs)})))
  (let [factory (model/tensor-factory model)
        es (sort-by first (seq model))]
    (loop [es es offset 0]
      (if-let [[k n] (first es)]
        (let [num-vals (int (apply * (:shape n)))
              vals (Arrays/copyOfRange xs offset (+ offset num-vals))]
          (tensors/copy-from-input! factory (:value n) (seq vals))
          (recur (next es) (+ offset num-vals)))
        model))))
