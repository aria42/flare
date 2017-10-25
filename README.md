# tensors

For no particular reason, a library for Tensor dataflow computations. Mostly for learning purposes. Current features

* Define dynamic tensor graphs (ala PyTorch or DynNet) or use the static graph approach (Tensorflow, Caffe2) for better performance.
* Supports basic tensor ops (sum, multiplication, concat, split, etc.) as well as bidirectional LSTMs and fixed embeddings.
* Currently only supports non-batch operations (working on auto-batching)
* Tensor implementation is pluggable, but best is currently [Neanderthal](http://github.com/uncomplicate/neanderthal) which supports Intel MKL or CUDA. 

## Examples

See some examples in the `src/tensors/examples` directory for usage examples.


## License

Copyright © 2017 aria42.com

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
