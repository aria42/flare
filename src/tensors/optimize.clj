(ns tensors.optimize
  (:require [tensors.core :as tensors]
            [plumbing.core :as p]
            [tensors.model :as model]
            [tensors.compute :as compute]))

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

(defn update-model! [optimizer model ^double batch-size ^double grad-clip]
  (doseq [[param-name param-node] (seq model)]
    ;; clip gradient
    (let [grad-clip (double grad-clip)
          normalizer (/ 1.0 batch-size)
          clip-min (- (Math/abs grad-clip))
          clip-max (Math/abs grad-clip)]
      (tensors/transform!
       (model/tensor-factory model)
       (p/safe-get param-node :grad)
       (fn ^double [^longs _ ^double x]
         (clip (* normalizer x) clip-min clip-max))))
    ;; perform update
    (let [state (p/safe-get (meta param-node) ::state)
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
      (tensors/transform! factory (p/safe-get param-node :grad) 0.0))))

(defrecord SGD [factory alpha]
  Optimizer
  ;; intentional no-op
  (init [this params-node] nil)
  (update-params! [this params-node state]
     (tensors/grad-step!
      factory
      (p/safe-get params-node :value)
      alpha
      (p/safe-get params-node :grad))
    nil))

(defrecord Adadelta [factory ^double eta ^double gamma ^double epsilon]
  Optimizer
  (init [this params-node]
    ;; sum-of-squares
    (let [shape (p/safe-get params-node :shape)]
      {:sum-sqs (tensors/zeros factory shape)
       :update (tensors/zeros factory shape)}))
  (update-params! [this params-node state]
    (let [g (p/safe-get params-node :grad)
          v (p/safe-get params-node :value)
          {:keys [sum-sqs, update]} state]
      ;; update gradient squared
      (tensors/transform! factory sum-sqs g
       (fn ^double [^longs _ ^double cur ^double gi]
         (+ (* gamma cur) (* (- 1.0 gamma) gi gi))))
      ;; copy gradient to update
      (tensors/copy-from-input! factory update g)
      ;; make update look like
      (tensors/transform! factory update sum-sqs
        (fn ^double [^longs _ ^double gi ^double G2i]
          (* gi (/ eta (Math/sqrt (+ epsilon G2i))))))
      ;; perform update
      (tensors/transform! factory v update
        (fn ^double [^longs _ ^double cur ^double u]
          (- cur u)))
      state)))



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
          (tensors/transform! factory (:grad n) 0.0))
        (loop [sum-loss 0.0 data data]
          (if-let [x (first data)]
            (when-let [g (build-graph x)]
              (let [n (compute/forward-pass! (compute/with-model-params model g) factory)
                    loss (first (:value n))]
                ;; accumulate gradient
                (assoc n :grad (tensors/copy-from-input! factory (:grad n) [1]))
                (compute/backward-pass! n)
                (recur (+ sum-loss loss) (next data))))
            [sum-loss (model/to-doubles model :grad)]))))))

(defn rand-bump-test [diff-fn ^doubles xs]
  (let [n (alength xs)
        rng #(- (* 2.0 (rand)) 1.0)
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
