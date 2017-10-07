(ns tensors.compute
  (:require [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [tensors.graph-ops :as go]
            [tensors.model :as model]
            [tensors.cache-pool :as cache-pool]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Compiled Graph Protocols + Operations

(s/defschema CompiledNode
  (assoc cg/Node
         :value s/Any
         :grad s/Any))

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

(s/defschema CompiledOpNode
  "Compiled operation has a tensor operation associated with
  with the CompiledNode as well as the graph operation definition"
  (merge CompiledNode
         cg/OpNode
         {:tensor-op TensorOp}))

(s/defschema CompiledRootNode
  (assoc CompiledNode
         :compiled? (s/eq true)
         ;; contains model for underlying parameters
         ;; used to fill values with model values
         ;; to allow for parameter sharing
         :model model/PModel))

(s/defn ensure-tensor-op
  "valdiates that tensor op valid for a computation,
   delegates down to `TensorOp` itself via `TensorFactory`"
  [factory :- tensors/PFactory
   result-node  :- cg/Node
   arg-nodes :- [cg/Node]]
  (let [op-key (-> result-node :graph-op cg/op-key)
        tensor-op (tensors/get-op factory op-key)]
    (ensure-valid?! tensor-op arg-nodes)
    tensor-op))

(defn ensure-tensor! [node key factory]
  (let [cache (-> factory meta :cache)
        t (cache-pool/get-obj cache (:shape node))
        return-fn #(cache-pool/return-obj cache (:shape node) t)]
    (tensors/fill! factory t 0.0)
    (-> node
        (assoc key t)
        (with-meta (merge (meta node) {[::return key] return-fn})))))

(defn release-tensor! [node key]  
  (when-let [return-fn (-> node meta (get [::return key]))]
    (return-fn))
  (dissoc node key))

(defn with-tensors [node factory model]
  (case (:type node)
    ;; must create a new vlaue
    :input (ensure-tensor! node :value factory)
    :constant (throw (ex-info "Not Supported"))
    ;; re-use the model values
    :params (model/canonical-node model (:ref-name node))
    ;; new values + grad
    :op (-> node (ensure-tensor! :value factory) (ensure-tensor! :grad factory))))

(defn with-release-tensors [node]
  (case (:type node)
    ;; must create a new vlaue
    :input (release-tensor! node :value)
    :constant (throw (ex-info "Not Supported"))
    ;; new values + grad
    :op (-> node (release-tensor! :value) (release-tensor! :grad))
    :params nil)
  node)

(defn with-tensor-op [node factory]
  (if (= (:type node) :op)
    (let [tensor-op (ensure-tensor-op factory node (:children node))]
      (assoc (prep tensor-op node)
             :tensor-op tensor-op))
    node))

(defn validate-graph! [node]
  (let [all-nodes (graph/post-order-nodes node)
        type->nodes (group-by :type all-nodes)
        inputs (:input type->nodes)
        params (:params type->nodes)
        op-nodes (:op type->nodes)
        name->op-nodes (group-by :ref-name op-nodes)]
    ;; ensure inputs are leaves
    (when-let [non-leaf-input (seq (filter (comp seq :children) inputs))]
      (throw (ex-info "Non-leaf input nodes" {:bad non-leaf-input})))
    ;; ensure params are leaves
    (when-let [non-leaf-params (seq (filter (comp seq :children) params))]
      (throw (ex-info "Non-leaf param nodes" {:bad non-leaf-params})))
    ;; ensure no duplicate names for nodes
    (when-let [duplicate (some #(> (count (val %)) 1) name->op-nodes)]
      (throw (ex-info "Op node names need to be unique"
                      {:duplicate duplicate})))))

(defn -compile-hack [node factory input->vals model]
  (let [node  (-> node
                  (with-tensors factory model)
                  (with-tensor-op factory))]
    (when (= :input (p/safe-get node :type))
      (let [vals (p/safe-get input->vals (:ref-name node))]
        (tensors/copy-from-input! factory (p/safe-get node :value) vals)))
    node))

(defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  [target model input->vals]
  (let [provided-keys (set (keys input->vals))
        nodes (graph/post-order-nodes target)
        factory (model/tensor-factory model)
        input->node (p/for-map [n nodes :when (= :input (:type n))]
                               (:ref-name n) n)
        existing-keys (set (keys input->vals))]
    ;; Ensure provided expected input values
    (when-let [missing (seq (set/difference existing-keys provided-keys))]
      (throw (ex-info "Missing input needed" {:missing missing})))
    ;; Copy input values to node tensors
    (graph/bottom-up-walk
     target
     (fn walk-fn [node]
       (let [node (-compile-hack node factory input->vals model)]
         (if-not (seq (:children node))
           ;; leaf node has no computation
           node
           ;; op node, fetch tensor-op
           ;; execute forward computation
           (let [tensor-op (:tensor-op node)]
             (p/safe-get node :value)
             (forward-node-pass! tensor-op node))))))))

(defn backward-pass-walk
  [node]
  (with-release-tensors
    (if-not (= :op (:type node))
      node
      (let [tensor-op (p/safe-get node :tensor-op)]
        (p/safe-get node :grad)
        (backward-node-pass! tensor-op node)))))

(s/defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes
   that have gradients (basically non-inputs) in graph"
  [target :- CompiledRootNode]
  (graph/top-down-walk target backward-pass-walk))
