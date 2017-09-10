(ns tensors.examples.logistic-regression
  (:require [tensors.neanderthal-ops :as no]
            [tensors.compute :as compute]
            [tensors.core :as tensors]
            [tensors.model :as model]
            [tensors.graph-ops :as go]
            [tensors.train :as train]
            [tensors.computation-graph :as cg]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.core :refer :all]))

(defn generate-data [num-datums num-classes num-feats]
  (let [r (java.util.Random. 0)
        W (dge num-classes num-feats)
        b (dv num-classes)]
    (alter! W (fn ^double [^long idx1 ^long idx2 ^double w] (.nextDouble r)))
    (alter! b (fn ^double [^long idx ^double w] (.nextDouble r)))
    (for [_ (range num-datums)]
      (let [rand-feats (mapv (fn [_] (if (.nextBoolean r) 1.0 0.0)) (range num-feats))
            f (dv rand-feats)
            class-idx (.nextInt r (int num-classes))
            activations (axpy (mv W f) b)
            label (imax activations)]
        {"f" rand-feats "label" [label]}))))

(defn batch-generator [])

(defn train []
  (let [num-classes 5
        num-feats 10
        factory (no/->Factory)
        m (model/simple-param-collection factory)
        W (model/add-params! m [num-classes num-feats] :name "W")
        b (model/add-params! m [num-classes] :name "b")
        feat-vec (go/strech (cg/input "f" [num-feats]) 1)
        activations (go/squeeze (go/+ (go/* W feat-vec) (go/strech b 1)) 1)
        ;; keep 1 as the "correct" label
        label (cg/input "label" [1])
        loss (go/cross-entropy-loss activations label)
        loss (compute/compile-graph loss factory m)
        data (generate-data 100 3 10)
        batch-gen #(partition 10 data)]
    (train/sgd! m loss batch-gen)
    ))

