(ns tensors.node
  (:require [clojure.string :as str]
            [schema.core :as s]
            [tensors.core :as tensors]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Datastructure

(defrecord Node
    [type
     shape
     ref-name
     value
     grad
     graph-op
     tensor-op
     children])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Datastructure

(def ^:dynamic *current-input-scope* [])

(defn ^String scoped-name
  "fully qualified node name with scopes"
  [^String node-name]
  (let [sb (StringBuilder.)]
    (doseq [x *current-input-scope*]
      (.append sb ^String x)
      (.append sb "/"))
    (.append sb node-name)
    (.toString sb)))

(defmacro with-scope [^String scope-name & body]
  `(binding [*current-input-scope*
             (conj *current-input-scope* (name ~scope-name))]
     ~@body))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Making Nodes

(s/defn input :- Node
  "Create input node. The intent is for the node to be re-used
   with different provided tensor values"
  ([input-name :- String shape :- tensors/Shape]
   (map->Node
    {:type :input
     :shape shape
     :ref-name (scoped-name input-name)}))
  ([shape :- tensors/Shape]
   (input (name (gensym "input")) shape)))

(s/defn constant :- Node
  "Create constant variable with provided tensor and factory.
  The tensor data needs to be able to used with tensors/from-nums"
  ([input-name :- String factory :- tensors/PFactory tensor-like :- s/Any]
   (let [t (tensors/from-nums factory tensor-like)]
     (map->Node
      {:type :constant
       :shape (tensors/shape factory t)
       :value t
       :ref-name (scoped-name input-name)})))
  ([factory :- tensors/PFactory tensor-like :- s/Any]
   (constant (name (gensym "input")) factory tensor-like)))

(defmacro definput [input-var shape]
  `(def ~input-var (input ~(name input-var) ~shape)))

(defmacro defparams [params-var shape]
  `(def ~params-var (params ~(name params-var) ~shape)))

(let [idx (java.util.concurrent.atomic.AtomicInteger.)]
  (defn gen-name [^String prefix]
    (str prefix ":" (.getAndIncrement idx))))
