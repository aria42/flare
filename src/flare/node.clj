(ns flare.node
  "Abstraction for a node in a computational graph. The most common thing
   to use here would be the `flare.node/const` function for returning a
   constant "
  (:require [clojure.string :as str]
            [flare.core :as flare])
  (:import [clojure.lang Keyword]
           [java.util.concurrent.atomic AtomicLong]))

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

(defmacro with-scope
  "everything in body gets the `scope` appended to default scope"
  [^String scope-name & body]
  `(binding [*current-input-scope*
             (conj *current-input-scope* (name ~scope-name))]
     ~@body))

(defmacro let-scope
  "Like `let` except the right-hand-side of each let binding
   is wrapped in `node/with-scope` using the string name of the
   left-hand-side variable. The goal is to properly nest the names
   of all nodes and parameters in graph

  ```clojure
     (let-scope
       [left (module/affine model 10 [5])
        right (module/affine model 10 [5])]
       (cg/+ left right))
  ```

  Then the model has `(\"left/W\",\"left/b\", \"right/W\", \"right/b\")`"
  [bindings & body]
  (let [new-bindings (->> bindings
                          (partition-all 2)
                          (mapcat (fn [[k v]]
                                    [k `(node/with-scope ~(name k) ~v)])))]
    `(let ~(vec new-bindings)
       ~@body)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Making Nodes

(defn ^Node const
  "A node which is a constant in a graph, typically used in dynamic graphs
  as leaves. The `data` is either a tensor or the default factory
  can make a tensor from it."
  ([^String input-name data]
   (let [t (if (satisfies? flare/Tensor data)
             data
             (flare/from (:factory (flare/state)) data))]
     (map->Node
      {:type :constant
       :shape (flare/shape t)
       :value t
       :ref-name (scoped-name input-name)})))
  ([data]
   (const (name (gensym "input")) data)))

(let [idx (AtomicLong.)]
  (defn gen-name
    [^String prefix]
    (str prefix ":" (.getAndIncrement idx))))
