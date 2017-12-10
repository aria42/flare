(ns flare.optimize
  (:require [flare.core :as flare]
            [flare.model :as model]
            [flare.compute :as compute]
            [flare.cache-pool :as cache-pool]))

(defprotocol Optimizer
  (init [this params-node]
    "return state required for future computations")
  (update-params! [this params-node state]
    "write to :value of params node and return updated state"))

(defn ^:static ^:private clip
  ^double [^double x ^double min ^double max]
  (if (> x max)
    max
    (if (< x min)
      min
      x)))

(defn batch-norm-clip-grad [model ^double batch-size ^double grad-clip]
  (let [grad-clip (double grad-clip)
        normalizer (/ 1.0 batch-size)
        clip-min (- (Math/abs grad-clip))
        clip-max (Math/abs grad-clip)]
    (doseq [[pname param-node] (seq model)]
      (flare/transform!
       (:grad param-node)
       (fn ^double [^double x]
         (clip (* normalizer x) clip-min clip-max))))))

(defn update-model! [optimizer model]
  (doseq [[param-name param-node] (seq model)]
    ;; perform update
    (let [state (::state (meta param-node))
          new-state (update-params! optimizer param-node state)]
      ;; update model with opt meta-data
      (model/with-metadata! model param-name ::state new-state))))

(defn init-model! [optimizer model]
  (doseq [[param-name param-node] (seq model)]
    (let [state (init optimizer param-node)]
      (model/with-metadata! model param-name ::state state))))

(defn reset-batch!
  "Reset all gradients except the target node"
  [optimizer model]
  ;; zero out gradients for param nodes
  (doseq [[_ param-node] model]
    (flare/transform! (:grad param-node) 0.0)))

(defrecord SGD [factory ^double alpha]
  Optimizer
  ;; intentional no-op
  (init [this params-node] nil)
  (update-params! [this params-node state]
    (flare/transform!
     (:value params-node)
     (:grad params-node)
     (fn ^double [^double cur ^double grad]
       (- cur (* alpha grad))))))

(defn update-expected-sqs [accum ^double gamma update]
  (flare/scale! accum gamma)
  (flare/add! accum (- 1.0 gamma) (flare/pow update 2.0)))

(defrecord Adadelta [factory ^double eta ^double gamma ^double epsilon]
  Optimizer
  (init [this params-node]
    ;; sum-of-squares
    (let [shape (:shape params-node)]
      {:exp-grad-sqs (flare/zeros factory shape :no-cache? true)
       :exp-delta-sqs (flare/zeros factory shape :no-cache? true)}))
  (update-params! [this params-node state]
    (let [g (:grad params-node)
          v (:value params-node)
          {:keys [exp-grad-sqs, exp-delta-sqs, delta]} state]
      ;; updaete gradient squared
      ;; E[g2_t] = gamma E[g2_{t-1}] + (1-gamma)  g2_t
      (update-expected-sqs exp-grad-sqs gamma g)
      (let [delta (flare/div exp-delta-sqs epsilon exp-grad-sqs epsilon)]
        (flare/pow! delta 0.5)
        (flare/mult! delta g)
        ;; actually move params
        (flare/add! v (- eta) delta)
        ;; perform update
        ;; E[d2_t] = gamma E[d2_{t-1}] + (1-gamma)  d2_t
        (update-expected-sqs exp-delta-sqs gamma delta))
      state)))

(defn ada-delta
  ([eta gamma epsilon]
   (->Adadelta (:factory (flare/state)) eta gamma epsilon))
  ([factory ^double eta ^double gamma ^double epsilon]
   (->Adadelta factory eta gamma epsilon)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; For Testing Bump Tests ala
;; https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/

(defprotocol DiffFn
  (dim [this])
  (val-at [this ^doubles xs] 
    "return [f(x) grad-f(x)] pair as double/double-array"))

(defn loss-fn [model build-graph data]
  (reify DiffFn
    (dim [this] (model/total-num-params model))
    (val-at [this xs]
      (let [factory (model/tensor-factory model)]
        ;; fill parameters with input
        (model/from-doubles! model xs)
        ;; clear gradients 
        (doseq [[_ n] model]
          (flare/transform! (:grad n) 0.0))
        (loop [sum-loss 0.0 data data]
          (if-let [x (first data)]
            (when-let [g (build-graph x)]
              (let [n (if (:eager? (flare/state))
                        g
                        (compute/forward-pass!
                         (compute/with-model-params model g) factory))
                    loss (first (:value n))]
                ;; accumulate gradient
                (assoc n :grad (flare/copy! (:grad n) [1]))
                (compute/backward-pass! n)
                (recur (+ sum-loss ^double loss) (next data))))
            [sum-loss (model/to-doubles model :grad)]))))))

(defn rand-bump-test [diff-fn ^doubles xs]
  (let [n (alength xs)
        rng #(- (* 2.0 (Math/random)) 1.0)
        dir (repeatedly (alength xs) rng)
        [fx grad] (val-at diff-fn xs)
        get-bump (fn [^double alpha]
                   (map (fn [^double x ^double d] (+ x (* alpha d))) xs dir))
        ^double expected (reduce + (map * dir grad))]
    (println "Start bump " [fx grad])
    (loop [eps 0.5]
      (let [[^double plus-x _] (val-at diff-fn (double-array (get-bump eps)))
            [^double neg-x _] (val-at diff-fn (double-array (get-bump (- eps))))
            approx (/ (- plus-x neg-x) (* 2 eps))
            delta (Math/abs (- approx expected))]
        (println {:approx approx :expected expected :plus-x plus-x :neg-x neg-x})
        (printf "At eps %.3f, delta is %.4f\n" eps delta)
        (when (> delta 0.0001)
          (if (< eps 1.0e-10)
            (throw (ex-info "Backtrack underflow" {:eps eps}))
            (recur (* eps 0.5))))))))
