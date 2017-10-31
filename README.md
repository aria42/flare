# tensors

A Clojure library for dynamic neural nets (e.g., PyTorch, DynNet). Mostly for learning purposes, but totally useable and pretty performant (see [Performance]("#perf") below). Current features:

* Define dynamic neural net graphs tensor (ala PyTorch or DynNet) or use the static graph approach (Tensorflow, Caffe2) for better performance.
* Supports basic tensor ops (sum, multiply, concat, split, etc.) as well as LSTM cells and fixed embeddings. 
* Currently only supports non-batch operations (working on auto-batching...)
* Tensor implementation is pluggable, but best is currently [Neanderthal](http://github.com/uncomplicate/neanderthal) which supports Intel MKL, CUDA, and OpenCL. Removed the ND4J implementation, since no clear reason to use given Neanderthal was faster on every operation.

## Examples

See some examples in the `src/tensors/examples` directory for usage examples. Single best entry is probably the LSTM example.


## License

Copyright © 2017 aria42.com

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
