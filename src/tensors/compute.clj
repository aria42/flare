(ns tensors.compute
  (:require [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.graph :as graph]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [tensors.graph-ops :as go]
            [tensors.model :as model]))

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
  (forward-node-pass! [this output! inputs]
    "compute the forward pass of the algorithm, for each node, compute
     `:value` tensor for passed in node, using the `:children` nodes
      and their `:value` tensors`")
  (backward-node-pass! [this output inputs!]
    "compute the `:grad` gradient tensor on each node reaching down to the leaves
     (which include the parameter nodes)"))

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
         :model model/PModel
         ;; cached mapping of input ref names
         ;; to underlying tensors to allow population
         :input->vals {s/Str s/Any}))

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

(defn with-tensors [node factory model]
  (case (:type node)
    ;; don't need a gradient for inputs
    :input  (assoc node :value (tensors/zeros factory (:shape node)))
    :params (model/canonical-node model (:ref-name node))
    :op     (assoc node
                   :value (tensors/zeros factory (:shape node))
                   :grad (tensors/zeros factory (:shape node)))))

(defn with-tensor-op [node factory children]
  (if (= (:type node) :op)
    (assoc node :tensor-op (ensure-tensor-op factory node children))
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

(s/defn compile-graph :- CompiledRootNode
  [target-node :- cg/Node
   factory :- tensors/PFactory
   model :- model/PModel]
  (validate-graph! target-node)
  (let [compile-walk (fn [node children]
                       (-> node
                           (with-tensors factory model)
                           (with-tensor-op factory children)
                           (assoc :children children)))
        compiled-target (graph/bottom-up-walk  target-node compile-walk)
        compiled-nodes (graph/post-order-nodes compiled-target)
        input->vals (p/for-map [n compiled-nodes :when (= :input(:type n))]
                               (:ref-name n) (:value n))]
    (assoc compiled-target
           :compiled? true
           :model model
           :input->vals input->vals)))

(s/defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  [target :- CompiledRootNode factory :- tensors/PFactory input->vals]
  (let [provided-keys (set (keys input->vals))
        existing-keys (set (keys (:input->vals target)))]
    ;; Ensure provided expected input values
    (when-let [missing (seq (set/difference existing-keys provided-keys))]
      (throw (ex-info "Missing input needed" {:missing missing})))
    ;; Copy input values to node tensors
    (doseq [[input-name input-node-val] (:input->vals target)
            :let [provided-vals (get input->vals input-name)]]
      (tensors/copy-from-input! factory input-node-val provided-vals))
    ;; Bottom up walk to compute forward values
    (graph/bottom-up-walk
     target
     (fn [node children]
       (if-not (seq children)
         ;; leaf node has no computation
         node
         ;; op node, fetch tensor-op
         ;; execute forward computation
         (let [tensor-op (:tensor-op node)]
           (p/safe-get node :value)
           (forward-node-pass! tensor-op node children)
           node))))
    ;; Return original node
    target))

(s/defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes"
  [target :- CompiledNode])
