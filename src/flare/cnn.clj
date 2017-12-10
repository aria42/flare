(ns flare.cnn
  (:require [clojure.spec.alpha :as s]
            [flare.module :as module]
            [flare.node :as node]
            [flare.computation-graph :as cg]
            [flare.core :as flare]))

(s/def ::kernel-spec
  (s/keys :req-un [::width ::height]
          :opt-un [::stride ::gap]))
(s/def ::width int?)
(s/def ::height int?)
(s/def ::stride int?)

(defn conv1D-builder
  "builds a function which takes a sequence of input nodes
   and returns the sequence of convolutional features over each window

   Input:
      model: `PModel` to add kernel params to
      inputs: seq of `n` input nodes
      kernel-spec: width, height, stride (defaults to 1) of kernel

   Outputs: Lazy sequence of size `(partition width stride inputs)` nodes
   each with dimension `height` from kernel heights"
  [model ^long input-dim kernel-spec]
  (s/conform ::kernel-spec kernel-spec)
  (let [{:keys [^long width, ^long height]} kernel-spec
        stride (get kernel-spec :stride 1)
        kernel-input-dim (* (long width) input-dim)
        kernel (module/affine model (long height) [kernel-input-dim]
                              :bias? false)]
    (fn [inputs]
      (when (< (count inputs) width)
        (throw (ex-info "Too small input for window"
                        {:input-size (count inputs)
                         :kernel-width width})))
      (for [window (partition width stride inputs)
            :let [x (apply cg/concat 0 window)]]
        (module/graph kernel x)))))

(defn cnn-1D-feats
  "module that builds max-pool conv features.

  The output is concattened height of all kernel-specs which
  has the max value of the conv-1D across the sliding windows"
  [model input-dim kernel-specs]
  (s/conform (s/coll-of ::kernel-spec) kernel-specs)
  (let [mk-conv (fn [kernel-spec]
                  (node/with-scope (format "conv-1D(%s)" kernel-spec)
                    (conv1D-builder model input-dim kernel-spec)))
        zero (flare/zeros (:factory (flare/state)) [input-dim])
        padding (node/const "cnn-padding" zero)
        ^long min-len (apply clojure.core/max (map :width kernel-specs))
        ;; make eager so all params created
        conv-fns (mapv mk-conv kernel-specs)]
    (when-not (distinct? kernel-specs)
      (throw (ex-info "Duplicate kernel-specs"
                      {:kernel-specs kernel-specs})))
    (reify module/PModule
      (graph [this inputs0]
        (let [num-padding-needed (max 0 (- min-len (count inputs0)))
              inputs (concat (repeat num-padding-needed padding) inputs0)]
          (->> conv-fns
               (map (fn [conv-fn]
                      (apply cg/max (conv-fn inputs))))
               (apply cg/concat 0)))))))
