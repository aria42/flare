(ns tensors.compute
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [plumbing.core :as p]
            [schema.core :as s]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;  Compiled Graph Protocols + Operations
;;;

(s/defschema CompiledNode
  (assoc graph/Node
         :value s/Any
         :grad s/Any))

(defprotocol TensorOp
  (^boolean valid? [this input-nodes])
  (^tensors/PFactory factory [this])
  (forward-pass [this output! inputs])
  (backward-pass [this output inputs!]))

(s/defschema CompiledOpNode
  "Compiled operation has a tensor operation associated with
  with the CompiledNode as well as the graph operation definition"
  (merge CompiledNode
         graph/OpNode
         {:tensor-op TensorOp}))

(defn bottom-up-walk [node walk-fn]
  (walk-fn node (map #(bottom-up-walk % walk-fn) (:children node))))

(defn post-order-nodes [target]
  (conj (vec (mapcat post-order-nodes (:children target))) target))

(defn compile-walk [node compiled-children factory]
  (-> node
      ;; always add value tensor
      (assoc :value (tensors/zeros factory (:shape node)))
      ;; add tensor op for graph ops
      (p/?>  (= :op (:type node))
             (assoc :tensor-op
                    (tensors/get-op factory (-> node :graph-op graph/op-key)))
      ;; add gradient for non-inputs
      (p/?> (not= :input (:type node))
            (assoc :grad (tensors/zeros factory (:shape node))))
      ;; add compiled children
      (assoc :children compiled-children))))

(defn validate-graph! [node]
  (let [all-nodes (post-order-nodes node)
        type->nodes (group-by :type all-nodes)
        inputs (:input type->nodes)
        params (:params type->nodes)
        name->nodes (group-by :ref-name all-nodes)]
    (when-let [duplicate (some #(> (count (val %)) 1) name->nodes)]
      (throw (RuntimeException.
              (str "Reference to multiple nodes " (key duplicate)))))
    (when-let [bad-input (some (comp seq :children) inputs)]
      (let [msg (str "Input needs to be leaf in graph: " (:ref-name bad-input))]
        (throw (RuntimeException. msg))))
    (when-let [bad-params (some (comp seq :children) params)]
      (let [msg (str "Params needs to be leaf in graph: " (:ref-name bad-params))]
        (throw (RuntimeException. msg))))))

(s/defn compile-graph! :- CompiledNode
  [target-node :- graph/Node
   factory :- tensors/PFactory]
  (validate-graph! target-node)
  (let [compiled-target (bottom-up-walk
                         target-node
                         (fn [node children]
                           (compile-walk node children factory)))
        compiled-nodes (post-order-nodes compiled-target)
        input->vals (p/for-map [n compiled-nodes :when (= :input(:type n))]
                               (:ref-name n) (:value n))]
    (assoc compiled-target
           :compiled? true
           :input->vals input->vals)))

(s/defn forward-pass!
  "forward-pass will topographic walk through graph writing to `:value`
  key on all compiled nodes. You can then look up and retrieve the tensors
  associated with any node"
  [target :- CompiledNode input->vals]
  )

(s/defn backward-pass!
  "backward-pass through all the parameter nodes associated with
   the graph computation, will write to `:grad` key for all nodes"
  [target :- CompiledNode])


(def lr
  (let [num-classes 2
        num-feats 10
        W (graph/params "W" [num-classes num-feats])
        b (graph/params "bias" [num-classes 1])
        feat-vec (graph/input "f" [num-feats 1])
        activations (graph/+ (graph/* W feat-vec) b)
        probs (graph/soft-max activations)
        label (graph/input "label" [1])
        loss (graph/cross-entropy-loss probs label)]
    loss))


(def simple-graph
  (let [X (graph/input "X" [2 2])
        Y (graph/input "Y" [2 2])]
    (graph/+ X Y)))
