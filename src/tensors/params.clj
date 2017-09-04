(ns tensors.params
  (:import [java.util HashMap Map])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.computation-graph :as cg]))

(defmethod cg/get-param-rng :uniform
  [{:keys [rand-seed, lower, upper]}]
  (let [lower (double (or lower -1.0))
        upper (double (or upper 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double []
      (+ lower (* (- upper lower) (.nextDouble r))))))

(defmethod cg/get-param-rng :normal
  [{:keys [rand-seed, mean, sigma]}]
  (let [mean (double (or mean 0.0))
        sigma (double (or sigma 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double []
      (/ (+ (.nextGaussian r) mean) sigma))))

(s/defn init-params
  [shape :- tensors/Shape
   get-param :- clojure.lang.IFn$D]
  (for [_ (range (first shape))]
    (if (= 1 (count shape))
      (get-param)
      (init-params (drop 1 shape) get-param))))
