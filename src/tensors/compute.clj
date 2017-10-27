(ns tensors.compute
  (:require [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [tensors.cache-pool :as cache-pool]
            [tensors.model :as model])
  (:import [tensors.node Node]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Compiled Graph Protocols + Operations

(defprotocol TensorOp
  (ensure-valid?! [this input-nodes]
    "Ensure the operation can be perfed with the tensor operation. Some
    imp0lemntations may support limited dimension or sizes")
  (forward-node-pass! [this node]
    "compute the forward pass of the algorithm, for each node, compute
     `:value` tensor for passed in node, using the `:children` nodes
     and their `:value` tensors. Returns the node in case any other
     computations are added to the node for use in the backward pass.")
  (backward-node-pass! [this node]
    "compute the `:grad` gradient tensor on each child of passed in node reaching
     down to the leaves  (which include the parameter nodes).
     Returns the node so that downstream backward-node-pass!
     calls can use added data."))


;; This is nowhere near ready for use yet
(defprotocol BatchTensorOp
  (batch-signature [this node]
    "signature that can be used to group operations")
  (batch-forward-node-pass! [this sig nodes]
    "compute the batch version of the forward pass")
  (batch-backward-node-pass! [this sig nodes]
    "compute the batch version of the backward pass"))

(defn -trivial-batch-forward-pass! [tensor-op nodes]
  (mapv
   (fn [n] (forward-node-pass! tensor-op n))
   nodes))

(defn -trivial-batch-backward-pass! [tensor-op nodes]
  (mapv
   (fn [n] (backward-node-pass! tensor-op n))
   nodes))

(defn node-tensor-op
  "valdiates that tensor op valid for a computation,
   delegates down to `TensorOp`"
  [factory ^Node result-node]
  (assert (identical? :op (.type result-node)))
  (let [op-key (-> result-node .graph-op cg/op-key)
        tensor-op (tensors/get-op factory op-key)]
    (ensure-valid?! tensor-op (.children result-node))
    tensor-op))

(defmacro key-case
  "like case, but only for keyword single arguments. Uses `identical?`
   for speed and avoids hashing. In micro-benchmarks, much faster.."
  [k & clauses]
  `(cond
     ~@(mapcat (fn [[ok# v#]] (list (list 'clojure.lang.Util/identical k ok#) v#))
               (partition 2 clauses))
     :else (throw (ex-info "No matching keyword" {:key ~k}))))

(defn return-key [key]
  (key-case key :value ::return-value :grad ::return-grad))

(def +zero+ (Double. 0.0))

(defn with-tensors [^Node node factory]
  (if (identical? (.type node) :op)
    (-> node
        (assoc :value (tensors/zeros factory (.shape node)))
        (assoc :grad (tensors/zeros factory (.shape node))))
    node))

(defn validate-input-keys [nodes ^java.util.Map input->vals]
  (let [provided (.keySet input->vals)]
    (doseq [^Node n nodes  :when (identical? :input (.type n))]
      (when-not (.contains provided (.ref-name n))
        (throw (ex-info "Missing required key" {:key (.ref-name n)}))))))

(defn with-model-params [model ^Node node]
  (let [factory (model/tensor-factory model)]
    (graph/bottom-up-walk
     node
     (fn [^Node n]
       (if (identical? (.type n) :params)
         (model/canonical-node model (.ref-name n))
         n)))))

(defn with-inputs! [factory node input->vals]
  (let [nodes (graph/topographic node)]
    (validate-input-keys nodes input->vals)
    (doseq [^Node n nodes]
      (when (identical? :input (.type n))
        (let [vals (p/safe-get input->vals (.ref-name n))]
          (assert (.value n))
          (tensors/copy! factory (.value n) vals))))))

(defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  [factory ^Node target]
  (let [nodes (graph/topographic target)
        computed-nodes (java.util.HashMap. (count nodes))
        get-canonical (fn [^Node node] (.get computed-nodes (.ref-name node)))]
    ;; Copy input values to node tensors
    (doseq [^Node onode nodes]
      (when-not (get-canonical onode)
        (let [children (.children onode)
              canonical-children (java.util.ArrayList. (count children))]
          (doseq [c children]
            (let [cc (get-canonical c)]
              (when-not cc
                (throw (ex-info "No child canonical" {:missing c})))
              (.add canonical-children cc)))
          (let [node (assoc onode :children canonical-children)
                ^Node node (with-tensors node factory)]
            (.put computed-nodes 
                  (.ref-name node)
                  (if (seq (.children node))
                    (forward-node-pass! (node-tensor-op factory node) node)
                    node))))))
    (.get computed-nodes (.ref-name target))))

(defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes
   that have gradients (basically non-inputs) in graph"
  [factory ^Node target]
  (let [nodes (reverse (graph/post-order-nodes target))]
    (doseq [^Node n nodes :when (identical? :op (.type n))]
      (backward-node-pass! (node-tensor-op factory n) n))))
