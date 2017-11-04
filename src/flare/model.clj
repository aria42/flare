(ns flare.model
  (:require [schema.core :as s]
            [flare.core :as flare]
            [flare.cache-pool :as cache-pool]
            [flare.node :as node]
            [plumbing.core :as p])
  (:import [java.util Arrays HashMap Map]
           [flare.node Node]
           [clojure.lang IFn$ODD]))

(s/defschema InitParamSpec
  "Spec for how to generate parameter entries independently
  implement `get-param-rng` multi-method for `:distribution`
  for a new distirbution"
  {:distribution s/Keyword
   s/Any s/Any})

(defmulti ^IFn$ODD get-param-rng :distribution)

(defmethod get-param-rng :uniform
  [{:keys [rand-seed, lower, upper]}]
  (let [lower (double (or lower -1.0))
        upper (double (or upper 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double [^longs indices ^double x]
      (+ lower (* (- upper lower) (.nextDouble r))))))

(defmethod get-param-rng :xavier-uniform
  [{:keys [rand-seed, ^long num-in, ^long num-out]}]
  (let [r (java.util.Random. (long (or rand-seed 0)))
        low (- (Math/sqrt (/ 6.0 (+ num-in num-out))))
        high (- low)]
    (fn ^double [^longs indices ^double x]
      (+ low (* (- high low) (.nextDouble r))))))

(defmethod get-param-rng :normal
  [{:keys [rand-seed, mean, sigma]}]
  (let [mean (double (or mean 0.0))
        sigma (double (or sigma 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double [^longs indices ^double x]
      (/ (+ (.nextGaussian r) mean) sigma))))

(defprotocol PModel
  "A model is effectively a paramaeter collection that holds graph nodes
   representing parameter groups.

   By convention, any implementation should be seqable and return a
   sequence of [param-name, param-node] pairs for the model"
  (tensor-factory [this]
    "return the underlying tensor factory (`PTensorFactory`) for the model")
  (-add-params! [this param-name shape init-spec]
    "add parameters to the model (mutably), returns a param graph node. Some
     argument defaulting happens below so this is the internal method,
    use `add-params!` as the entry point")
  (-add-param-metadata! [this param-name key val]
    "add meta-data [key, val] pair to the node for the given `param-name`
     node. Useful for storing optimization data. Internal method due to some
     parameter defaulting.")
  (canonical-node [this param-name]
    "returns a caonical `Node` for the parameter (meaning it stores
      the \"true\" value/gradient for for the parameters)."))

(defn total-num-params [model]
  (p/sum (fn [[_ p]] (apply * (:shape p))) model))

(defprotocol -TestPModel
  "only for testing with models"
  (fix-param! [this param-name value]))

(defn add-params!
  "Add parameters of givne `shape` to model. Accepts some options
     * `:name` String name for parameter for inspection later
     * `:init` a map satisfying `InitParamSpec` for 
       distribution of how to sample values"
  [model shape & {:keys [name, init]}]
  (let [name (or name (clojure.core/name (gensym "param")))
        init (or init {:distribution :uniform})]
    (s/validate s/Str name)
    (s/validate InitParamSpec init)
    (-add-params! model name shape init)))

(defn with-metadata!
  "add a key-value pair to the metadata on the underlying
   parameter node. As name implies, `param-node-or-name` can
   be either current canonical node or the `Node.ref-name`
   for the parameter group. "
  [model param-node-or-name key value]
  (let [node-name (if (instance? Node param-node-or-name)
                    (.ref-name ^Node param-node-or-name)
                    param-node-or-name)]
    (-add-param-metadata! model node-name key value)))

(defn rand-params!
  "using the initialziation spec for each param group, generate
   a fresh setting of the parameters"
  [model]
  (doseq [[_ node] (seq model)]
    (when-let [init (:init node)]
      (flare/transform!
       (tensor-factory model)
       (p/safe-get  node :value)
       (get-param-rng init)))))

(s/defn simple-param-collection :- PModel
  "Simple collection of parameters
   NOTE: The meta-data of the param-collection gives you access
   to the underlying data. Don't use it except for an emergency!"
  ([] (simple-param-collection (:factory (flare/state))))
  ([factory :- flare/PTensorFactory]
   (let [m (java.util.HashMap.)]
     (with-meta
       (reify

         PModel
         (tensor-factory [this] factory)
         (-add-params! [this param-name shape init-spec]
           (let [param-name (node/scoped-name param-name)]
             (when-let [existing (.get m param-name)]
               (throw (ex-info "Existing param key" {:existing existing})))
             (let [node (node/-with-eager
                         (node/map->Node
                          {:type :params
                           :ref-name param-name
                           :value (flare/zeros factory shape)
                           :grad (flare/zeros factory shape)
                           :shape shape
                           :init init-spec})
                         factory)
                   ;; to determinize initializatin
                   init-spec (assoc init-spec :rand-seed (hash param-name))
                   get-param-val (get-param-rng init-spec)]
               ;; initialize param vals from init-spec
               (flare/transform! factory (:value node) get-param-val)
               (.put m param-name node)
               node)))
         (-add-param-metadata! [this param-name key val]
           (let [n (canonical-node this param-name)
                 n (with-meta n (assoc (meta n) key val))]
             (.put m param-name n)
             n))
         (canonical-node [this param-or-name]
           (if-let [v (.get m param-or-name)]
             v
             (throw (ex-info "Non-existtant param" {:key param-or-name}))))

         -TestPModel
         (fix-param! [this param-or-name tensor-data]
           (let [param-name (if (instance? Node param-or-name)
                              (.ref-name ^Node param-or-name)
                              param-or-name)]

             (when-let [param (.get m param-name)]
               (let [param-tensor (:value param)]
                 (flare/copy! factory param-tensor tensor-data)))))

         clojure.lang.Seqable
         (seq [this]
           (for [e m] [(key e) (val e)])))
       {:data m}))))


(defn ^doubles to-doubles
  "Flatten parameters into a single JVM double array. Can take
   either parameter values or current gradient, depending on `key`"
  ([model] (to-doubles model :value))
  ([model key]
   (when-not (#{:grad :value} key)
     (throw (ex-info "Key must be {:grad, :value}" {:key key})))
   (let [factory (tensor-factory model)
         xs (double-array (total-num-params model))
         es (sort-by first (seq model))]
     (loop [es es offset 0]
       (if-let [e (first es)]
         (let [[k n] e
               v (flatten (seq (get n key)))]
           (System/arraycopy
            (double-array v)
            0
            xs
            offset
            (count v))
           (recur (next es) (+ offset (count v))))
         xs)))))

(defn from-doubles!
  "Flatten parameters into a single vector and writes to each
   parameter group tensor `:value`"
  [model ^doubles xs]
  (when-not (= (alength xs) (total-num-params model))
    (throw (ex-info "Array doesn't match model size"
                    {:model-size (total-num-params model)
                     :array-len (alength xs)})))
  (let [factory (tensor-factory model)
        es (sort-by first (seq model))]
    (loop [es es offset 0]
      (if-let [[k n] (first es)]
        (let [num-vals (int (apply * (:shape n)))
              vals (Arrays/copyOfRange xs offset (+ offset num-vals))]
          (flare/copy! factory (:value n) (seq vals))
          (recur (next es) (+ offset num-vals)))
        model))))
