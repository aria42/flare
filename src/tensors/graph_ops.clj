(ns tensors.graph-ops
  (:refer-clojure :exclude [+ * concat])
  (:require
   [tensors.core :as tensors]
   [tensors.computation-graph :as cg]
   [schema.core :as s]
   [tensors.model :as model]
   [plumbing.core :as p]))

