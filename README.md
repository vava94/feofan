# feofan
A framework for TensorRT applications.

Inspired by https://en.wikipedia.org/wiki/Neuromonakh_Feofan.

## Opportunities:
* Running multiple neural networks
* Support for ".onnx", ".uff" and TRT ".engine" files

## Requirements:
* PC with NVIDIA GPU
* CUDA Toolkit installed
* TensorRT installed
* \[\[maby_unused\]\] Brain

This project uses third party functions to parse NN output. For example, the code uses a dynamic link to the [neural-adapter](https://github.com/vava94/neural-adapter) functions (you can disable it with preprocessor flags and set your own).

### P.S.
Please rerfer to https://github.com/vava94/feofan if using.


Using Cuda Font utility from [jetson-utils](https://github.com/dusty-nv/jetson-utils)