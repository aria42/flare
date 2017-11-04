(ns flare.compute
  (:require [flare.computation-graph :as cg]
            [flare.core :as flare]
            [flare.graph :as graph]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [flare.cache-pool :as cache-pool]
            [flare.model :as model])
  (:import [flare.node Node]
           [java.util LinkedList]
           [java.util.concurrent.atomic AtomicLong]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Compiled Graph Protocols + Operations


;; This is nowhere near ready for use yet
(defprotocol BatchTensorOp
  (batch-signature [this node]
    "signature that can be used to group operations")
  (batch-forward-node-pass! [this sig nodes]
    "compute the batch version of the forward pass")
  (batch-backward-node-pass! [this sig nodes]
    "compute the batch version of the backward pass"))

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

(defn with-inputs! 
  ([factory node input->vals]
   (let [nodes (graph/topographic node)]
     (validate-input-keys nodes input->vals)
     (doseq [^Node n nodes]
       (when (identical? :input (.type n))
         (let [vals (p/safe-get input->vals (.ref-name n))]
           (assert (.value n))
           (flare/copy! factory (.value n) vals))))))
  ([node input->vals]
   (with-inputs! (:factory (flare/state)) node input->vals)))

(defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  ([factory ^Node target]
   (forward-pass! factory target nil))
  ([factory ^Node target cache]
   (let [nodes (graph/topographic target)
         computed-nodes (java.util.HashMap.)
         get-canonical (fn [^Node node] (.get computed-nodes (.ref-name node)))]
     ;; Copy input values to node tensors
     (doseq [^Node onode nodes]
       (when-not (get-canonical onode)
         (let [children (.children onode)
               canonical-children (java.util.ArrayList.)]
           (doseq [c children]
             (let [cc (get-canonical c)]
               (when-not cc
                 (throw (ex-info "No child canonical" {:missing c})))
               (.add canonical-children cc)))
           (let [^Node node (assoc onode :children canonical-children)]
             (.put computed-nodes
                   (.ref-name node)
                   (if (identical? (.type node) :op)
                     (cg/-forward node factory cache)
                     node))))))
     (.get computed-nodes (.ref-name target))))
  ([node]
   (forward-pass! (p/safe-get (flare/state) :factory) node)))

(defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes
   that have gradients (basically non-inputs) in graph"
  ([factory ^Node target cache]
   (let [nodes (reverse (graph/post-order-nodes target))
         perf-map (-> factory meta (get-in [:debug :perf]))]
     (doseq [^Node n nodes :when (identical? :op (.type n))]
       (let [start (System/nanoTime)
             tensor-op (cg/-tensor-op n factory)
             node (cg/backward-node-pass! tensor-op  n)
             end (System/nanoTime)
             op-key (-> n :graph-op cg/op-key)
             sum-nanos (get-in perf-map [op-key :backward])]
         (when sum-nanos
           (.getAndAdd ^AtomicLong sum-nanos (- end start))))
       (when cache
         (cache-pool/return-obj cache (.shape n) (.value n))
         (cache-pool/return-obj cache (.shape n) (.grad n))))))
  ([factory ^Node target]
   (backward-pass! factory target nil)))

(defn free-tensors!
  ([node cache]
   (let [nodes (reverse (graph/post-order-nodes node))]
     (doseq [^Node n nodes :when (identical? :op (.type n))]
       (when (identical? (.type n) :op)
         (cache-pool/return-obj cache (.shape n) (.value n))
         (cache-pool/return-obj cache (.shape n) (.grad n))))))
  ([node] (free-tensors! node (:cache (flare/state)))))
