(ns flare.graph
  (:require [clojure.string :as str])
  (:import [java.util HashMap ArrayList LinkedList HashSet Stack]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Graph Walks

(defn bottom-up-walk [node walk-fn]
  ;; must `doall` for children since `walk-fn` can have side-effects
  (if-let [cs (seq (:children node))]
    (walk-fn (assoc node :children (mapv #(bottom-up-walk % walk-fn) cs)))
    (walk-fn node)))

(defn topographic [node]
  (let [ret (ArrayList.)
        stack (Stack.)
        marks (HashMap.)
        push-onto-stack (fn [x]
                          (let [m (get marks (:ref-name x) :none)]
                            (case m
                              :visited nil
                              :processing (throw (ex-info "Not a DAG"
                                                          {:cause :cyclic-graph
                                                           :node x}))
                              :none (.push stack x))))]
    (push-onto-stack node)
    (loop []
      (if-not (empty? stack)
        (let [n (.peek stack)
              n-name (:ref-name n)
              children (:children n)]
          (do
            (.put marks n-name :processing)
            (doseq [c children] (push-onto-stack c))
            (when (every? #(= (get marks (:ref-name %)) :visited) children)
              (.pop stack)
              (.add ret n)
              (.put marks n-name :visited))
            (recur)))
        ret))))

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
