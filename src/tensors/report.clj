(ns tensors.report
  (:refer-clojure :exclude [concat])
  (:require [schema.core :as s]
            [tensors.model :as model]
            [plumbing.core :as p]
            [tensors.core :as tensors]))

(s/defschema BatchInfo
  {:batch-loss s/Num
   :model model/PModel
   :batch [s/Any]})

(defprotocol Reporter
  (update! [this info])
  (clear! [this])
  (gen [this]))

(defn test-accuracy [get-data get-pred-node]
  (reify Reporter
    (update! [this info])
    (clear! [this])
    (gen [this]
      (let [[num-correct total]
            (->> (get-data)
                 (map (fn [[x label]]
                        (= label (-> x get-pred-node :value first))))
                 (reduce (fn [[num-correct total] correct?]
                           [(if correct? (inc num-correct) num-correct)
                            (inc total)])
                         [0 0]))]
        {:test-accuracy {:acc (/ (double num-correct) total) :n total}}))))

(defn avg-loss []
  (let [sum (atom 0.0)
        n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (swap! sum + (p/safe-get info :batch-loss))
        (swap! n + (count (p/safe-get info :batch))))
      (gen [this]
        {:avg-loss {:avg (/ @sum @n) :n @n}})
      (clear! [this]
        (reset! sum 0.0)
        (reset! n 0)))))

(defn grad-size []
  (let [sum-l2-norm (atom 0.0)
        sum-max-l1-norm (atom 0.0)
        n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (let [factory (model/tensor-factory (p/safe-get info :model))
              grads (mapcat (fn [[_ x]] (flatten (tensors/->clj factory (:grad x))))
                            (p/safe-get info :model))
              l2-norm (Math/sqrt (reduce (fn [res x] (+ res (* x x))) 0.0 grads))
              max-l1-norm (apply max (map (fn [x] (Math/abs (double x))) grads))]
          (swap! sum-l2-norm + l2-norm)
          (swap! sum-max-l1-norm + max-l1-norm)
          (swap! n inc)))
      (gen [this]
        {:l2-norm {:avg (/ @sum-l2-norm @n) :n @n}
         :max-l1-norm {:avg (/ @sum-max-l1-norm @n) :n @n}})
      (clear! [this]
        (reset! sum-l2-norm 0.0)
        (reset! sum-max-l1-norm 0.0)
        (reset! n 0)))))

(defn every [freq r]
  (let [n (atom 0)]
    (reify
      Reporter
      (update! [this info]
        (update! r info))
      (gen [this]
        (swap! n inc)
        (when (zero? (mod @n freq))
          (gen r)))
      (clear! [this]
        (reset! n 0)
        (clear! r)))))

(defn custom-value [key value-fn]
  (let [sum (atom 0.0)
        n (atom 0)]
    (reify
      Reporter
      (update! [this _]
        (swap! sum + (value-fn))
        (swap! n inc))
      (gen [this]
        {key {:total @sum :n @n}})
      (clear! [this]
        (reset! sum 0.0)
        (reset! n 0)))))

(defn concat [& reporters]
  (reify
    Reporter
    (update! [this info]
      (doseq [r reporters] (update! r info)))
    (gen [this]
      (reduce merge (map gen reporters)))
    (clear! [this]
      (doseq [r reporters] (clear! r)))))

(defn training
  ([] (concat (avg-loss) (grad-size)))
  ([freq]
   (every freq (training))))
