(ns tensors.module
  (:refer-clojure :exclude [comp])
  (:require [tensors.node :as node]
            [tensors.model :as model]
            [tensors.computation-graph :as cg]
            [tensors.compute :as compute]
            [tensors.cache-pool :as cache-pool]
            [tensors.core :as tensors])
  (:import [tensors.node Node]))

(defprotocol PModule
  "A module knows ow to construct a graph for a given input. This is
   the work-horse for dynamic graph construction."
  (graph
    [this input]
    [this input1 input2]
    "build a graph for any input(s)"))

(defn predict-fn
  "on computed graph, also apply an `arg-max` operation and
   return the forward value of that node and converts predicted
   tensor to clj data. If the value is scalar, unwraps and returns number

   If module doesn't build graph for input, prediciton returns nil.


   Will create a cache for this prediction fn to ensure
   predictions cache and re-use empty tensors in forward pass"
  [factory score-module]
  (let [cache (compute/cache factory 100)]
    (fn [& inputs]
      (when-let [n (apply graph score-module inputs)]
        (let [forward (compute/forward-pass! factory (cg/arg-max n) cache)
              predict-val (-> forward :value seq)]
          (when cache
            (compute/free-tensors! forward cache))
          (if (tensors/scalar-shape? (:shape forward))
            (first predict-val)
            predict-val))))))

(defn from-op [op]
  (when-not (satisfies? cg/GraphOp op)
    (throw (ex-info "Must be graph-op" {:op op})))
  (reify
    PModule
    (graph [this input]
      (cg/add-graph-op op [input]))
    (graph [this input1 input2]
      (cg/add-graph-op op [input1 input2]))))

(defn comp [& ms]
  (reify
    PModule
    (graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))))


(defn affine
  "graph: returns single output (Wx + b)
   params: [W b]"
  [model num-out in-shape]
  (let [[first-dim & rest] in-shape
        out-shape (vec (cons num-out rest))
        W (model/add-params! model [num-out first-dim]
                             :name "W"
                             :init {:distribution :xavier-uniform
                                    :num-in first-dim
                                    :num-out num-out})
        b (model/add-params! model out-shape
                             :name "b"
                             :init {:distribution :normal})]
    (reify
      PModule
      (graph [this x]
        (cg/+ (cg/* W x) b)))))

(defn static
  "Module with a static graph, the input is used to just input values
   (this is similar to TensorFlow's `feed-dict` and it has the nice
   performance proprties of static graphs.)"
  [factory root-node]
  (reify PModule
    (graph [this input->vals]
      (compute/with-inputs! factory root-node input->vals)
      root-node)))
