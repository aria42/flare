<img src="http://aria42.com/images/flare.png" alt="aria42/flare logo" title="flare logo" align="right" width="250" />

# flare

[Codox Link](http://aria42.com/flare/)

A Clojure library for dynamic neural nets (e.g., PyTorch, DynNet). Mostly for learning purposes, but totally usable and pretty performant (see Performance below). See introductory blog post [here](http://aria42.com/blog/2017/11/Flare-Clojure-Neural-Net). Current features:

* Define dynamic neural net graphs tensor (ala PyTorch or DynNet) or use the static graph approach (Tensorflow, Caffe2) for better performance.
* Supports basic tensor ops (sum, multiply, concat, split, etc.) as well as LSTM cells, 1-dimensional CNNs (for NLP applications) and fixed embeddings.
* Currently only supports non-batch operations (working on auto-batching...)
* Tensor implementation is pluggable, but best is currently [Neanderthal](http://github.com/uncomplicate/neanderthal) which supports Intel MKL, CUDA, and OpenCL. Removed the ND4J implementation, since no clear reason to use given Neanderthal was faster on every operation.

## Installation

At the moment, the only supported tensor backend is via Neanderthal, which currently only uses [Intel MKL](https://software.intel.com/en-us/mkl). You'll need to install this library (although if you're doing a lot of neural net work locally, you probably already have). Note that if you're on a Mac, you'll need to disable [System Integrity Protection](https://www.imore.com/el-capitan-system-integrity-protection-helps-keep-malware-away), in order for Neanderthal to load the native libs from the `DYLD_LIBRARY_PATH` environment variable.

## Examples

See some examples in the [`src/flare/examples`](https://github.com/aria42/flare/tree/master/src/flare/examples) directory for usage examples. Single best entry is probably the LSTM example. But here are some simple ones:

```clojure
(ns example
  (:require [flare.core :as flare]
            [flare.node :as node]
            [flare.computation-graph :as cg]))

(flare/init!)
;; Z = X + Y example
(def X (node/const [2 3]))
(def Y (node/const [3 4]))
(def Z (cg/+ X Y))
(:value Z)
;;
returns [5.0 7.0] tensor
```

Notice that the computation happens *eagerly*, you don't need to call `forward` on any node, the operation happens as soon great the graph node. The numerical computations use a pluggable backend, but these can use native hardware. While the above example is slightly verbose compared to PyTorch, for longer pieces of code you get more expressiveness. Here's an example of building a simple bidirectional LSTM sentiment classifier for a given sentence. It uses the concept of a *module* which is a protocol for building graphs given an input. Typically, you can make modules that close over new parameters and other modules in a module:

```clojure
(defn lstm-sent-classifier [model word-emb lstm-size num-classes]
  (node/let-scope
      ;; let-scope so the parameters get smart nesting of names
      [emb-size (embeddings/embedding-size word-emb)
       num-dirs 2
       input-size (* num-dirs emb-size)
       hidden-size (* num-dirs lstm-size)
       lstm (rnn/lstm-cell model input-size hidden-size)
       hidden->logits (module/affine model num-classes [hidden-size])]
    (reify
      module/PModule
      (graph [this sent]
        ;; build logits
        (let [inputs (embeddings/sent-nodes word-emb sent)
              [outputs _] (rnn/build-seq lstm inputs (= num-dirs 2))
              train? (:train? (meta this))
              hidden (last outputs)
              hidden (if train? (cg/dropout 0.5 hidden) hidden)]
          (module/graph hidden->logits hidden))))))
```

## Performance

The big surprise is that Flare is relatively performant so far, no slower than PyTorch overall and for many cases 2x-3x faster than PyTorch. The actual tensor computations both utilize Intel MPK so differences in performance mostly stem from using Clojure vs Python for graph construction.

## To Do

* GPU Support: Neanderthal supports many platforms here, so this should be straightforward
* Auto-Batching: I like the auto-batching idea, described in this [paper](https://arxiv.org/abs/1705.07860). I have the start of a auto-batching computation, which shows some sign of speeding up training substantially. I'm not quite happy with the design, and want to take some time to think through if there's a cleaner way to add this.

## License

Copyright © 2017 aria42.com

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
