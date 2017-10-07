(ns tensors.node)

(defrecord Node
    [type shape ref-name value grad graph-op tensor-op children])
