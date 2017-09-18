(ns tensors.graph-ops
  (:refer-clojure :exclude [+ *])
  (:require
   [tensors.core :as tensors]
   [tensors.computation-graph :as cg]
   [schema.core :as s]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  Graph Edge Operations

(defrecord SumGraphOp []
  cg/GraphOp
  (op-key [this] :+)
  (op-validate! [this input-nodes]
    (when (empty? input-nodes)
      (throw (ex-info "Has empty inputs" {:inputs []})))
    (let [[shape & the-rest :as shapes] (map :shape input-nodes)]
      (when-not (every? #(= shape %) the-rest)
        (throw (ex-info "Not all input shpaes match"
                        {:shapes (vec shapes)})))))
  (forward-shape [this input-nodes]
    (:shape (first input-nodes)))
  (op-descriptor [this] "+"))


(defrecord MultGraphOp []
  cg/GraphOp
  (op-key [this] :*)
  (op-validate! [this input-nodes]
    (when (empty? input-nodes)
      (throw (ex-info "Need at least one input"
                      {:num-args (count input-nodes)})))
    (let [shapes (map :shape input-nodes)]
      (when-let [bad (some (fn [s] (> (count s) 2)) shapes)]
        (throw (ex-info "Only max 2d allowed" {:bad bad})))
      (when-not (every?
                 (fn [[[a b] [c d]]] (= b c))
                 (partition 2 shapes))
        (throw (ex-info "Incompatibl shapes" {:shapes shapes})))))
  (forward-shape [this input-nodes]
    (let [shapes (map :shape input-nodes)
          num-rows (ffirst shapes)]
      (if-let [num-cols (second (last shapes))]
        [num-rows num-cols]
        [num-rows])))
  (op-descriptor [this] "*"))

(s/defn ensure-vector-tensor?! [prefix shape :- tensors/Shape]
  (let [n (count shape)
        dim-counts (frequencies shape)]
    (when-not (and (= n 2) (= (get dim-counts 1) (dec n)))
      (throw (ex-info (str prefix ": Must be row or column vector")
                      {:dim-counts dim-counts})))))

(defrecord SqueezeGraphOp [dim-to-squeeze]
  cg/GraphOp
  (op-key [this] :squeeze)
  (op-descriptor [this] (str "squeeze-" dim-to-squeeze))
  (op-validate! [this nodes]
    (when-not (= 1 (count nodes))
      (throw (ex-info "Squeeze only takes single input"
                      {:num-args (count nodes)})))
    (let [shape (vec (:shape (first nodes)))]
      (when-not (= 1 (nth shape dim-to-squeeze))
        (throw (ex-info "Squeeze at ``dimension`` not 1"
                        {:dim-to-squeeze dim-to-squeeze
                         :shape shape})))
      (into (subvec shape 0 dim-to-squeeze)
            (subvec shape (inc dim-to-squeeze)))))
  (forward-shape [this inputs]
    (let [shape (vec (:shape (first inputs)))]
      (into (subvec shape 0 dim-to-squeeze)
            (subvec shape (inc dim-to-squeeze))))))

(defrecord StrechGraphOp [dim-to-insert]
  cg/GraphOp
  (op-key [this] :strech)
  (op-descriptor [this] (str "strech-" dim-to-insert))
  (op-validate! [this inputs]
    (when-not (= 1 (count inputs))
      (throw (ex-info "Stretch  only takes single input"
                      {:causes inputs}))))
  (forward-shape [this inputs]
    (let [shape (vec (:shape (first inputs)))]
      (vec
       (concat (subvec shape 0 dim-to-insert)
               [1]
               (when (< (inc dim-to-insert) (count shape))
                 (subvec shape (inc dim-to-insert))))))))

(defrecord CrossEntropyLossOp []
  cg/GraphOp
  (op-key [this] :cross-entropy-loss)
  (op-descriptor [this] "cross-entropy-loss")
  (op-validate! [this [activations label & other]]
    (when (or (nil? activations) (nil? label) (seq other))
      (throw (ex-info "Expect (activations, labels) pair"
                      {:activations activations :label label})))
    (when-not (tensors/scalar-shape? (:shape label))
      (throw (ex-info "Label sjhould be effectively scalar"
                      {:label-shape (:shape label)})))
    (when-not (tensors/vector-shape? (:shape activations))
      (throw (ex-info "Activations should be vector"
                      {:activations-shape (:shape activations)}))))
  (forward-shape [this [activations label :as input-nodes]]
    [1]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Public Graph Operations


(defn + [& inputs]
  (cg/add-graph-op (SumGraphOp.) inputs))

(defn * [& inputs]
  (cg/add-graph-op (MultGraphOp.) inputs))

(defn cross-entropy-loss [activations label]
  (cg/add-graph-op (CrossEntropyLossOp.) [activations label]))

(defn squeeze [input dim-to-squeeze]
  (cg/add-graph-op (SqueezeGraphOp. dim-to-squeeze) [input]))

(defn strech [input dim-to-insert]
  (cg/add-graph-op (StrechGraphOp. dim-to-insert) [input]))
