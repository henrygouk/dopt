dopt
====

A numerical optimisation and deep learning framework for D.

Current features include:

* Ability to construct symbolic representations of tensor-valued functions
* Basic arithmetic and mathematical operations (add, sub, mul, div, abs, log, exp, ...)
* Basic matrix operations (multiplication, transpose)
* Reverse-mode automatic differentiation
* Stochastic gradient descent
* Neural network primitives
* Neural network construction utilities
* Framework to add third party operations and their derivatives, and the ability register implementations for both the CPU and CUDA backends.

The project is still in the early stages, and some things might not work properly yet. Some planned future features include:

* The ability to add optimisation passes to the CPU and CUDA backends.
* More sophisticated online optimisation algorithms (SGD+momentum, AdaGrad, ADAM, etc)

Docs
----

Documentation can be found [here](https://henrygouk.github.io/dopt/).

Using
-----

The easiest way to use dopt is by adding it as a dependency in your project's dub configuration file. See dub's [getting started page](http://code.dlang.org/getting_started) for more information about how to do this.

Example
-------

See `examples/mnist.d` for the full code.

```
void main(string[] args)
{
    import dopt.core;
    import dopt.nnet;
    import dopt.online;

    auto data = loadMNIST(args[1]);

    auto features = float32([100, 1, 28, 28]);
    auto labels = float32([100, 10]);

    auto layers = dataSource(features)
                    .convolutional(32, [5, 5])
                    .relu()
                    .maxPool([2, 2])
                    .convolutional(32, [5, 5])
                    .relu()
                    .maxPool([2, 2])
                    .dense(10)
                    .softmax();

    auto network = new NeuralNetwork([layers, layers.crossEntropy(dataSource(labels))]);

    auto updater = sgd(network.loss, network.parameters);

    import std.range : zip;

    foreach(fs, ls; zip(data.trainFeatures.chunks(100), data.trainLabels.chunks(100)))
    {
        auto loss = updater([
            features: Buffer(fs.joiner().array()),
            labels: Buffer(ls.joiner().array())
        ]);

        import std.stdio;

        writeln(loss);
    }
}
```
