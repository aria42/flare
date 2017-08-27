(ns tensors.compute
  (:require [tensors.graph :as graph]
            [tensors.core :as tensors]
            [plumbing.core :as p]
            [schema.core :as s]
            [clojure.set :as set]
            [tensors.compute :as compute]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Compiled Graph Protocols + Operations

(s/defschema CompiledNode
  (assoc graph/Node
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
         graph/OpNode
         {:tensor-op TensorOp}))

(defn bottom-up-walk [node walk-fn]
  (walk-fn node (map #(bottom-up-walk % walk-fn) (:children node))))

(defn post-order-nodes [target]
  (conj (vec (mapcat post-order-nodes (:children target))) target))

(defn ensure-tensor-op
  [factory node children]
  (let [op-key (-> node :graph-op graph/op-key)
        tensor-op (tensors/get-op factory op-key)]
    (ensure-valid?! tensor-op children)
    tensor-op))

(defn compile-walk [node compiled-children factory]
  (-> node
      ;; always add value tensor
      (assoc :value (tensors/zeros factory (:shape node)))
      ;; add tensor op for graph ops
      (p/?>  (= :op (:type node))
             (assoc :tensor-op (ensure-tensor-op factory node compiled-children)
                    )
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
  [target :- CompiledNode factory :- tensors/PFactory input->vals]
  (let [input-nodes (:input (group-by :type (post-order-nodes target)))
        provided-input-keys (set (keys input->vals))
        existing-input-keys (set (map :ref-name input-nodes))]
    ;; Ensure provided expected input values
    (when-let [missing (seq (set/difference existing-input-keys provided-input-keys ))]
      (throw (RuntimeException. (str "Missing needed inputs: " missing))))
    ;; Copy input values to node tensors
    (doseq [{:keys [value, ref-name]} input-nodes]
      (tensors/copy-from-input! factory value (get input->vals ref-name)))
    ;; Bottom up walk to compute forward values
    (bottom-up-walk
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


(def lr
  (let [num-classes 2
        num-feats 3
        W (graph/input "W" [num-classes num-feats])
        b (graph/strech (graph/input "bias" [num-classes]) 1)
        feat-vec (graph/strech (graph/input "f" [num-feats]) 1)
        activations (graph/squeeze (graph/+ (graph/* W feat-vec) b) 1)
        probs (graph/soft-max activations)
        label (graph/input "label" [1])
        loss (graph/cross-entropy-loss probs label)]
    {:loss loss
     :activations activations}))


(def simple-graph
  (let [X (graph/input "X" [2 2])
        Y (graph/input "Y" [2 2])
        Z (graph/input "Z" [2 2])]
    (graph/* Z (graph/+ X Y))))
