(ns flare.node
  (:require [clojure.string :as str]
            [schema.core :as s]
            [flare.core :as flare])
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
       (cg/+ left right))```


  Then the model will have `(\"left/W\",\"left/b\", \"right/W\", \"right/b\")`"
  [bindings & body]
  (let [new-bindings (->> bindings
                          (partition-all 2)
                          (mapcat (fn [[k v]]
                                    [k `(node/with-scope ~(name k) ~v)])))]
    `(let ~(vec new-bindings)
         ~@body)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Making Nodes

(s/defn input :- Node
  "Create a Node intended to be re-used as input for a static graph.
    Initialized to zeros"
  ([factory :- flare/PTensorFactory input-name :- String shape :- flare/Shape]
   (map->Node
    {:type :input
     :shape shape
     :value (flare/zeros factory shape)
     :ref-name (scoped-name input-name)}))
  ([input-name :- String shape :- flare/Shape]
   (input @flare/*factory input-name shape))
  ([shape :- flare/Shape]
   (input (name (gensym "input")) shape)))

(s/defn constant :- Node
  "A node which is a constant in a graph, typically used in dynamic graphs as leaves.
  The `data` is meant to be anything `flare.core/from` would accept"
  ([factory :- flare/PTensorFactory input-name :- String data]
   (let [t (flare/from factory data)]
     (map->Node
      {:type :constant
       :shape (flare/shape factory t)
       :value t
       :ref-name (scoped-name input-name)})))
  ([input-name :- String data]
   (constant @flare/*factory input-name data))
  ([data]
   (constant (name (gensym "input")) data)))

(let [idx (java.util.concurrent.atomic.AtomicInteger.)]
  (defn gen-name [^String prefix]
    (str prefix ":" (.getAndIncrement idx))))
