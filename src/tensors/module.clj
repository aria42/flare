(ns tensors.module
  (:refer-clojure :exclude [comp])
  (:require [tensors.node :as node]
            [tensors.model :as model]
            [tensors.computation-graph :as cg]
            [tensors.compute :as compute]
            [tensors.cache-pool :as cache-pool])
  (:import [tensors.node Node]))

(defprotocol Module
  "A module knows ow to construct a graph for a given input. This is
   the work-horse for dynamic graph construction.

   By convention, impls should also implemnt `IFunction` with
   the `graph` behavior"
  (graph
    [this input]
    [this input1 input2]
    "build a graph for any input(s)"))

(defn forward!
  "computes forward pass for the graph constructed by the module
  on the provided inputs"
  [factory module & inputs]
  (let [n (apply graph module inputs)]
    (compute/forward-pass! factory n)))

(defn predict
  "on computed graph, also apply an `arg-max` operation and
   return the forward value of that node"
  [factory score-module & inputs]
  (when-let [n (apply graph score-module inputs)]
    (let [cache (-> score-module meta :cache)
          forward (compute/forward-pass! factory (cg/arg-max n) cache)
          predict-val (-> forward :value seq)]
      (when cache
        (compute/free-tensors! forward cache))
      predict-val)))

(defn from-op [op]
  (when-not (satisfies? cg/GraphOp op)
    (throw (ex-info "Must be graph-op" {:op op})))
  (reify 
    Module
    (graph [this input]
      (cg/add-graph-op op [input]))
    (graph [this input1 input2]
      (cg/add-graph-op op [input1 input2]))

    clojure.lang.IFn
    (invoke [this input]
      (graph this input))
    (invoke [this input1 input2]
      (graph this input1 input2))))

(defn comp [& ms]
  (reify
    Module
    (graph [this input]
      (reduce
       (fn [result m] (graph m result))
       input
       (reverse ms)))

    clojure.lang.IFn
    (invoke [this input] (graph this input))))


(defn affine
  "graph: returns single output (Wx + b)
   params: [W b]"
  [model num-out in-shape]
  (node/with-scope "affine"
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
        Module
        (graph [this x]
          (cg/+ (cg/* W x) b))
        clojure.lang.IFn
        (invoke [this x] (graph this x))))))
