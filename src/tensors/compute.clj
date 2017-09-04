(ns tensors.compute
  (:require [tensors.computation-graph :as cg]
            [tensors.core :as tensors]
            [tensors.model :as model]
            [tensors.graph :as graph]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [tensors.graph-ops :as go]
            [tensors.compute :as compute]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Compiled Graph Protocols + Operations

(s/defschema CompiledNode
  (assoc cg/Node
         :value s/Any
         :grad s/Any))

(defprotocol TensorOp
  (ensure-valid?! [this input-nodes])
  (forward-node-pass! [this output! inputs])
  (backward-node-pass! [this output inputs!]))

(s/defschema CompiledOpNode
  "Compiled operation has a tensor operation associated with
  with the CompiledNode as well as the graph operation definition"
  (merge CompiledNode
         cg/OpNode
         {:tensor-op TensorOp}))

(defn ensure-tensor-op
  [factory node children]
  (let [op-key (-> node :graph-op cg/op-key)
        tensor-op (tensors/get-op factory op-key)]
    (ensure-valid?! tensor-op children)
    tensor-op))

(defn compile-walk [node children factory]
  (-> node
      ;; always add value tensor
      (assoc :value (tensors/zeros factory (:shape node)))
      ;; add tensor op for graph ops
      (p/?>  (= :op (:type node))
             (assoc :tensor-op (ensure-tensor-op factory node children)
                    )
      ;; add gradient for non-inputs
      (p/?> (not= :input (:type node))
            (assoc :grad (tensors/zeros factory (:shape node))))
      ;; add compiled children
      (assoc :children children))))

(defn validate-graph! [node]
  (let [all-nodes (graph/post-order-nodes node)
        type->nodes (group-by :type all-nodes)
        inputs (:input type->nodes)
        params (:params type->nodes)
        name->nodes (group-by :ref-name all-nodes)]
    (when-let [duplicate (some #(> (count (val %)) 1) name->nodes)]
      (throw (RuntimeException.
              (str "Reference to multiple nodes " (key duplicate)))))
    (when-let [bad-input (some (comp seq :children) inputs)]
      (throw (ex-info "Input needs to be leaf in graph"
                      {:bad-ref-name bad-input})))
    (when-let [bad-params (some (comp seq :children) params)]
      (throw (ex-info "Params needs to be leaf in graph"
                      {:bad-ref-name bad-params})))))

(s/defn compile-graph! :- CompiledNode
  [target-node :- cg/Node
   factory :- tensors/PFactory]
  (validate-graph! target-node)
  (let [compiled-target (graph/bottom-up-walk
                         target-node
                         (fn [node children]
                           (compile-walk node children factory)))
        compiled-nodes (graph/post-order-nodes compiled-target)
        input->vals (p/for-map [n compiled-nodes :when (= :input(:type n))]
                               (:ref-name n) (:value n))]
    (assoc compiled-target
           :compiled? true
           :input->vals input->vals)))

(s/defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  [target :- CompiledNode factory :- tensors/PFactory input->vals]
  (let [input-nodes (:input (group-by :type (graph/post-order-nodes target)))
        provided-keys (set (keys input->vals))
        existing-keys (set (map :ref-name input-nodes))]
    ;; Ensure provided expected input values
    (when-let [missing (seq (set/difference existing-keys provided-keys))]
      (throw (ex-info "Missing input needed" {:missing missing})))
    ;; Copy input values to node tensors
    (doseq [{:keys [value, ref-name]} input-nodes]
      (tensors/copy-from-input! factory value (get input->vals ref-name)))
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
           (forward-node-pass! tensor-op node children)
           node))))
    ;; Return original node
    target))

(s/defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes"
  [target :- CompiledNode])


(comment 
  (def lr
    (let [num-classes 2
          num-feats 3
          m (model/simple-param-collection)
          W (model/add-params! m "W" [num-classes num-feats] {:type :normal})
          b (model/add-params! m "bias" [num-feats] {:type :normal})
          feat-vec (go/strech (cg/input "f" [num-feats]) 1)
          activations (go/squeeze (go/+ (go/* W feat-vec) b) 1)
          probs (go/soft-max activations)
          label (cg/input "label" [1])
          loss (go/cross-entropy-loss probs label)]
      {:loss loss
       :activations activations}))


  (def simple-graph
    (let [X (graph/input "X" [2 2])
          Y (graph/input "Y" [2 2])
          Z (graph/input "Z" [2 2])]
      (graph/* Z (graph/+ X Y)))))
