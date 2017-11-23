(ns flare.optimize
  (:require [flare.core :as flare]
            [flare.model :as model]
            [flare.compute :as compute]))

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
       (model/tensor-factory model)
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
  (let [factory (model/tensor-factory model)]
    (doseq [[_ param-node] model]
      (flare/transform! factory (:grad param-node) 0.0))))

(defrecord SGD [factory ^double alpha]
  Optimizer
  ;; intentional no-op
  (init [this params-node] nil)
  (update-params! [this params-node state]
    (flare/transform! 
     factory 
     (:value params-node)
     (:grad params-node)
     (fn ^double [^double cur ^double grad]
       (- cur (* alpha grad))))))

(defrecord Adadelta [factory ^double eta ^double gamma ^double epsilon]
  Optimizer
  (init [this params-node]
    ;; sum-of-squares
    (let [shape (:shape params-node)]
      {:exp-grad-sqs (flare/zeros factory shape)
       :exp-delta-sqs (flare/zeros factory shape)
       :delta (flare/zeros factory shape)}))
  (update-params! [this params-node state]
    (let [g (:grad params-node)
          v (:value params-node)
          {:keys [exp-grad-sqs, exp-delta-sqs, delta]} state]
      ;; updaete gradient squared
      ;; E[g2_t] = gamma E[g2_{t-1}] + (1-gamma)  g2_t
      (flare/transform! factory exp-grad-sqs g
       (fn ^double [^double cur ^double gi]
         (+ (* gamma cur) (* (- 1.0 gamma) gi gi))))
      ;; copy gradient to update
      (flare/copy! factory delta g)
      ;; make update look like
      (flare/transform! factory delta exp-grad-sqs
        (fn ^double [^double gi ^double G2i]
          (/ gi (Math/sqrt (+ epsilon G2i)))))
      (flare/transform! factory delta exp-delta-sqs
        (fn ^double [^double gi ^double d2i]
          (* gi (Math/sqrt (+ epsilon d2i)))))
      ;; perform update
      (flare/transform! factory v delta
        (fn ^double [^double cur ^double u]
          (- cur (* eta u))))
      (flare/transform! factory exp-delta-sqs delta
         (fn ^double [^double cur ^double di]
           (+ (* gamma cur) (* (- 1.0 gamma) di di))))
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
          (flare/transform! factory (:grad n) 0.0))
        (loop [sum-loss 0.0 data data]
          (if-let [x (first data)]
            (when-let [g (build-graph x)]
              (let [n (if (:eager? (flare/state))
                        g
                        (compute/forward-pass! (compute/with-model-params model g) factory))
                    loss (first (:value n))]
                ;; accumulate gradient
                (assoc n :grad (flare/copy! factory (:grad n) [1]))
                (compute/backward-pass! n)
                (recur (+ sum-loss ^double loss) (next data))))
            [sum-loss (model/to-doubles model :grad)]))))))

(defn rand-bump-test [diff-fn ^doubles xs]
  (let [n (alength xs)
        rng #(- (* 2.0 (Math/random)) 1.0)
        dir (repeatedly (alength xs) rng)
        [fx grad] (val-at diff-fn xs)
        get-bump (fn [alpha] (map (fn [x d] (+ x (* alpha d))) xs dir))
        expected (reduce + (map * dir grad))]
    (println "Start bump " [fx grad])
    (loop [eps 0.5]
      (let [[plus-x _] (val-at diff-fn (double-array (get-bump eps)))
            [neg-x _] (val-at diff-fn (double-array (get-bump (- eps))))
            approx (/ (- plus-x neg-x) (* 2 eps))
            delta (Math/abs (- approx expected))]
        (println {:approx approx :expected expected :plus-x plus-x :neg-x neg-x})
        (printf "At eps %.3f, delta is %.4f\n" eps delta)
        (when (> delta 0.0001)
          (if (< eps 1.0e-10)
            (throw (ex-info "Backtrack underflow" {:eps eps}))
            (recur (* eps 0.5))))))))
