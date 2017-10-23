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
  (prep [this node]
    "add any tensor fields useful to share between forward/backwward calls")
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

(s/defn ensure-tensor-op
  "valdiates that tensor op valid for a computation,
   delegates down to `TensorOp` itself via `TensorFactory`"
  [factory :- tensors/PFactory
   result-node  :- Node
   arg-nodes :- [Node]]
  (let [op-key (-> result-node .graph-op cg/op-key)
        tensor-op (tensors/get-op factory op-key)]
    (ensure-valid?! tensor-op arg-nodes)
    tensor-op))

(defmacro key-case [k & clauses]
  `(cond
     ~@(mapcat (fn [[ok# v#]] (list (list 'clojure.lang.Util/identical k ok#) v#))
               (partition 2 clauses))
     :else (throw (ex-info "No matching keyword" {:key ~k}))))

(defn return-key [key]
  (key-case key :value ::return-value :grad ::return-grad))

(def +zero+ (Double. 0.0))

(defn ensure-tensor! [^Node node key factory]
  (if-let [cache (-> factory meta :cache)]
    (let [[t return-fn] (cache-pool/get-obj cache (.shape node))]
      (when (identical? key :grad)
        (tensors/transform! factory t +zero+))
      (-> node
          (assoc key t)
          (with-meta (assoc (meta node) (return-key key) return-fn))))
    (assoc node key (tensors/zeros factory (.shape node)))))

(defn release-tensor! [node key]
  (when-let [return-fn (-> node meta (get (return-key key)))]
    (return-fn (:value node)))
  (dissoc node key))

(defn with-tensors [^Node node factory]
  (key-case (.type node)
            ;; must create a new vlaue
            :input (ensure-tensor! node :value factory)
            :constant node
            :params node
            ;; new values + grad
            :op (-> node
                    (ensure-tensor! :value factory)
                    (ensure-tensor! :grad factory))))

(defn with-tensor-op [^Node node factory]
  (if (identical? (.type node) :op)
    (let [tensor-op (ensure-tensor-op factory node (.children node))]
      (assoc (prep tensor-op node)
             :tensor-op tensor-op))
    node))

(defn -compile-hack [^Node node factory input->vals]
  (let [^Node node  (-> node
                        (with-tensors factory)
                        (with-tensor-op factory))]
    (when (identical? :input (.type node))
      (let [vals (p/safe-get input->vals (.ref-name node))]
        (assert (.value node))
        (tensors/copy-from-input! factory (.value node) vals)))
    node))

(defn validate-input-keys [nodes ^java.util.Map input->vals]
  (let [provided (.keySet input->vals)]
    (doseq [^Node n nodes  :when (identical? :input (.type n))]
      (when-not (.contains provided (.ref-name n))
        (throw (ex-info "Missing required key" {:key (.ref-name n)}))))))

(defn -forward-intrnal [^Node node]
  (if-not (seq (.children node))
    ;; leaf node has no computation
    node
    ;; op node, fetch tensor-op
    ;; execute forward computation
    (let [tensor-op (.tensor-op node)
          forward-node (forward-node-pass! tensor-op node)]
      forward-node)))

(defn with-model-params [model ^Node node]
  (let [factory (model/tensor-factory model)]
    (graph/bottom-up-walk
     node
     (fn [^Node n]
       (if (identical? (.type n) :params)
         (model/canonical-node model (.ref-name n))
         n)))))

(defn with-inputs [factory node input->vals]
  (let [nodes (graph/topographic node)]
    (validate-input-keys nodes input->vals)
    (doseq [^Node n nodes]
      (when (identical? :input (.type n))
        (let [vals (p/safe-get input->vals (.ref-name n))]
          (assert (.value n))
          (tensors/copy-from-input! factory (.value n) vals))))))

(defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  ([^Node target factory] (forward-pass! target factory {}))
  ([^Node target factory input->vals]
   (let [nodes (graph/topographic target)
         computed-nodes (java.util.HashMap. (count nodes))
         get-canonical (fn [^Node node] (.get computed-nodes (.ref-name node)))]
     (validate-input-keys nodes input->vals)
     ;; Copy input values to node tensors
     (doseq [^Node onode nodes]
       (when-not (get-canonical onode)
         (let [new-children (java.util.ArrayList. (count (:children onode)))]
           (doseq [c (:children onode)]
             (let [cc (get-canonical c)]
               (when-not cc
                 (throw (ex-info "No child canonical" {:missing c})))
               (.add new-children cc)))
           (let [node (assoc onode :children new-children)
                 ^Node node (-compile-hack node factory input->vals)
                 ^Node node (-forward-intrnal node)]
             (.put computed-nodes (.ref-name node) node)))))
     (.get computed-nodes (.ref-name target)))))

(defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes
   that have gradients (basically non-inputs) in graph"
  [target]
  (let [nodes (reverse (graph/post-order-nodes target))]
    (doseq [^Node n nodes :when (identical? :op (.type n))]
      (backward-node-pass! (.tensor-op n) n))
    (doseq [^Node n nodes]
      (key-case (.type n)
                ;; must create a new vlaue
                :input (release-tensor! n :value)
                :constant nil
                :params nil
                ;; new values + grad
                :op (do
                      (release-tensor! n :value)
                      (release-tensor! n :grad))))))
