(ns tensors.model
  (:import [java.util HashMap Map])
  (:require [schema.core :as s]
            [tensors.core :as tensors]
            [tensors.graph :as graph]))

(defprotocol PParameterCollection
  (add-params! [this param-name shape init-spec])
  (get-params [this param-name]))

(defmulti get-param-rng :type)

(defmethod get-param-rng :uniform
  [{:keys [rand-seed, lower, upper]}]
  (let [lower (double (or lower -1.0))
        upper (double (or upper 1.0))
        r (java.util.Random. (long (or rand-seed 0)))]
    (fn ^double []
      (+ lower (* (- upper lower) (.nextDouble r))))))


(s/defn init-params
  [shape :- tensors/Shape
   get-param :- clojure.lang.IFn$D]
  (for [_ (range (first shape))]
    (if (= 1 (count shape))
      (get-param)
      (init-params (drop 1 shape) get-param))))


(s/defrecord SimpleParamCollection
    [factory :- tensors/PFactory
     param-name->node :- java.util.HashMap]
  PParameterCollection
  (get-params [this param-name]
    (.get param-name->node param-name))
  (add-params! [this param-name shape init-spec]
    (let [param-name (graph/full-node-name param-name)
          get-param (get-param-rng init-spec)
          init-vals (init-params shape get-param)
          params {:type :params
                  :shape shape
                  :value (tensors/from-nums factory init-vals)
                  :grad (tensors/zeros factory shape)
                  :ref-name param-name}]
      (.put param-name->node param-name params)
      params)))


(s/defn simple-param-collection [factory :- tensors/PFactory]
  (SimpleParamCollection. factory (java.util.HashMap.)))
