(ns tensors.node
  (:require [clojure.string :as str]
            [schema.core :as s]
            [tensors.core :as tensors])
  (:import [clojure.lang Keyword]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Datastructure

(defrecord Node
    [type
     shape
     ref-name
     value
     grad
     graph-op
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
  "Create a Node intended to be re-used as input for a static graph.
    Initialized to zeros"
  ([factory :- tensors/PFactory input-name :- String shape :- tensors/Shape]
   (map->Node
    {:type :input
     :shape shape
     :value (tensors/zeros factory shape)
     :ref-name (scoped-name input-name)}))
  ([input-name :- String shape :- tensors/Shape]
   (input @tensors/*factory* input-name shape))
  ([shape :- tensors/Shape]
   (input (name (gensym "input")) shape)))

(s/defn constant :- Node
  "A node which is a constant in a graph, typically used in dynamic graphs as leaves.
  The `data` is meant to be anything `tensors.core/from` would accept"
  ([factory :- tensors/PFactory input-name :- String data]
   (let [t (tensors/from factory data)]
     (map->Node
      {:type :constant
       :shape (tensors/shape factory t)
       :value t
       :ref-name (scoped-name input-name)})))
  ([input-name :- String data]
   (constant @tensors/*factory* input-name data))
  ([data]
   (constant (name (gensym "input")) data)))

(let [idx (java.util.concurrent.atomic.AtomicInteger.)]
  (defn gen-name [^String prefix]
    (str prefix ":" (.getAndIncrement idx))))
