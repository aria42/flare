(ns flare.module
  (:refer-clojure :exclude [comp])
  (:require [flare.node :as node]
            [flare.model :as model]
            [flare.computation-graph :as cg]
            [flare.compute :as compute]
            [flare.cache-pool :as cache-pool]
            [flare.core :as flare]
            [flare.module :as module])
  (:import [flare.node Node]))

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
  ([score-module]
   (fn [& inputs]
     (when-let [n (apply graph score-module inputs)]
       (let [{:keys [factory, cache, eager?]} (flare/state)
             forward (if eager?
                       (cg/arg-max n)
                       (compute/forward-pass! factory (cg/arg-max n) cache))
             predict-val (-> forward :value seq)]
         (when cache
           (compute/free-tensors! forward cache))
         (if (flare/scalar-shape? (:shape forward))
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
  [model num-out in-shape
   & {:keys [bias?] :or {bias? true}}]
  (let [[first-dim & rest] in-shape
        out-shape (vec (cons num-out rest))
        sigma (/ 1.0 (Math/sqrt (double num-out)))
        init {:distribution :uniform :lower (- sigma) :upper sigma}
        W (model/add-params! model [num-out first-dim]
                             :name "W"
                             :init init)
        b (when bias?
            (model/add-params! model out-shape
                               :name "b"
                               :init init))]
    (reify
      PModule
      (graph [this x]
        (let [y (cg/* W x)]
          (if b (cg/+ y b) y))))))

(def l2-squared-loss
  (reify
    PModule
    (graph [this x y]
      (let [diff (cg/+ x (cg/scale -1.0 y))]
        (cg/sum-elems (cg/hadamard diff diff))))
    clojure.lang.IFn
    (invoke [this x y]
      (module/graph this x y))))
