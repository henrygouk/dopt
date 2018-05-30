dopt
====

[![DUB](https://img.shields.io/dub/v/dopt.svg)](http://code.dlang.org/packages/dopt)
[![Travis-CI](https://travis-ci.org/henrygouk/dopt.svg?branch=master)]()

A numerical optimisation and deep learning framework for D.

Current features include:

* Ability to construct symbolic representations of tensor-valued functions
* Basic arithmetic and mathematical operations (add, sub, mul, div, abs, log, exp, ...)
* Basic matrix operations (multiplication, transpose)
* Reverse-mode automatic differentiation
* Neural network primitives
* Neural network construction utilities
* Framework to add third party operations and their derivatives, and the ability register implementations for both the CPU and CUDA backends
* Online optimisation algorithms: SGD, ADAM, AMSGrad, and more to come!

The project is still in the early stages, and some things might not work properly yet. Some planned future features include:

* The ability to add optimisation passes to the CPU and CUDA backends
* More utilities for training deep networks (data loaders, standard training loops, etc)

Docs
----

Documentation can be found [here](https://henrygouk.github.io/dopt/). A brief outline of how to use this framework for deep learning is provided [here](https://henrygouk.github.io/dopt/dopt/nnet.html).

Using
-----

The easiest way to use dopt is by adding it as a dependency in your project's dub configuration file. See dub's [getting started page](http://code.dlang.org/getting_started) for more information about how to do this.

If you want to take advantage of the CUDA backend (currently required for most neural network operations) then you should also ensure that the `cuda` configuration is used. This is the sort of thing you would end up putting in your dub.json file:

```json
"dependencies": {
    "dopt": "~>0.3.11"
},
"subConfigurations": {
    "dopt": "cuda"
}
```

Example
-------

Examples for training networks on MNIST, CIFAR-10, CIFAR-100, and [SINS-10](https://cs.waikato.ac.nz/~ml/sins10/) are given in the `examples/` folder.