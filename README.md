dopt
====

[![DUB](https://img.shields.io/dub/v/dopt.svg)]()

A numerical optimisation and deep learning framework for D.

Current features include:

* Ability to construct symbolic representations of tensor-valued functions
* Basic arithmetic and mathematical operations (add, sub, mul, div, abs, log, exp, ...)
* Basic matrix operations (multiplication, transpose)
* Reverse-mode automatic differentiation
* Stochastic gradient descent
* Neural network primitives
* Neural network construction utilities
* Framework to add third party operations and their derivatives, and the ability register implementations for both the CPU and CUDA backends
* Online optimisation algorithms: SGD and ADAM

The project is still in the early stages, and some things might not work properly yet. Some planned future features include:

* The ability to add optimisation passes to the CPU and CUDA backends.
* More online optimisation algorithms (SGD+momentum, AdaGrad, etc)

Docs
----

Documentation can be found [here](https://henrygouk.github.io/dopt/).

Using
-----

The easiest way to use dopt is by adding it as a dependency in your project's dub configuration file. See dub's [getting started page](http://code.dlang.org/getting_started) for more information about how to do this.

Example
-------

Examples for training networks on MNIST and CIFAR10 are given in the `examples/` folder.