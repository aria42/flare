(ns flare.graph
  (:require [schema.core :as s]
            [clojure.string :as str]
            [plumbing.core :as p])
  (:import [java.util HashMap ArrayList LinkedList HashSet]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Walks

(defn bottom-up-walk [node walk-fn]
  ;; must `doall` for children since `walk-fn` can have side-effects
  (if-let [cs (seq (:children node))]
    (walk-fn (assoc node :children (mapv #(bottom-up-walk % walk-fn) cs)))
    (walk-fn node)))

(defn topographic [node]
  (let [marks (HashMap.)
        ret (ArrayList.)
        visit (fn visit [n]
                (let [m (get marks (:ref-name n) :none)]
                  (case m
                    :permanent nil
                    :temporary (throw (ex-info "Not a DAG"))
                    :none (do
                            (.put marks (:ref-name n) :temporary)
                            (doseq [c (:children n)]
                              (visit c))
                            (.put marks (:ref-name n) :permanent)
                            (.add ret n)))))]
    (visit node)
    ret))

(defn top-down-walk [node walk-fn]
  ;; walk-fn can update children so do a let-binding
  (let [node (walk-fn node)]
    (if-let [cs (seq (:children node))]
      (assoc node :children (mapv #(top-down-walk % walk-fn) cs))
      node)))

(defn post-order-nodes [target]
  (let [list (ArrayList.)
        queue (LinkedList.)
        seen? (HashSet.)
        add-to-queue (fn [x]
                       (when-not (.contains seen? (:ref-name x))
                         (.add queue x)
                         (.add seen? (:ref-name x))))]
    (add-to-queue target)
    (loop []
      (if-let [n (.poll queue)]
        (do
          (.add list n)
          (doseq [c (:children n)]
            (add-to-queue c))
          (recur))
        (reverse list)))))

(defn gen-binary [^long depth]
  (if (zero? depth)
    {:value (name (gensym "node"))}
    {:value (name (gensym "node"))
     :children [(gen-binary (dec depth)) (gen-binary (dec depth))]}))
