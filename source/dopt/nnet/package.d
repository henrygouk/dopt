/**
    This package contains a neural network framework backed by the automatic differentiation capabilities of dopt.

    $(UL
        $(LI $(D dopt.nnet) contains the $(D NeuralNetwork) class, which can be used to assist in the construction of
        DAG structured neural networks using operations defined in $(D dopt.core).)
        $(LI $(D dopt.nnet.layers) contains functions for constructing common layer types.)
        $(LI $(D dopt.nnet.paraminit) contains several different schemes for initialising network parameters.)
    )

    Examples:
    --------------------
    void main(string[] args)
    {
        import dopt.core;
        import dopt.nnet;
        import dopt.online;

        //Load the MNIST data---this function can be found in examples/mnist.d
        auto data = loadMNIST(args[1]);

        //Create variables representing the inputs to our network
        auto features = float32([100, 1, 28, 28]);
        auto labels = float32([100, 10]);

        //This chunk of code takes advantage of D's uniform function call syntax to chain some layers together
        auto layers = dataSource(features)
                     .convolutional(32, [5, 5])
                     .relu()
                     .maxPool([2, 2])
                     .convolutional(32, [5, 5])
                     .relu()
                     .maxPool([2, 2])
                     .dense(10)
                     .softmax();

        //Use the NeuralNetwork helper class to organise all the parameters
        auto network = new NeuralNetwork([layers, layers.crossEntropy(dataSource(labels))]);

        //Create an updater using one of the algorithms in dopt.online
        auto updater = sgd(network.loss, cast(Operation[])network.parameters);

        //Train the network using the data we loaded earlier
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
    --------------------

    Authors: Henry Gouk
*/
module dopt.nnet;

import std.algorithm : map, joiner;
import std.array : array;

import dopt.core;

public
{
    import dopt.nnet.layers;
    import dopt.nnet.paraminit;
}

/**
    This class provides an simple way to construct neural networks that can be trained with the $(D dopt.online)
    package.
*/
class NeuralNetwork
{
    public
    {
        /**
            Construct a neural network with a single output layer

            Params:
                output = The output layer for the new neural network.
        */
        this(const(Layer) output)
        {
            this([output]);
        }

        /**
            Construct a neural network with multiple output layers

            Params:
                outputs = The output layers for the new neural networks.
        */
        this(const(Layer)[] outputs)
        {
            mOutputs = outputs
                      .map!(x => x.expression)
                      .array();

            //Toposort the layers
            const(Layer)[] sortedLayers;
            bool[const(Layer)] visited;

            void toposort(const(Layer) l)
            {
                if(visited.get(l, false))
                {
                    return;
                }

                visited[l] = true;

                sortedLayers ~= l;

                foreach(c; l.deps)
                {
                    toposort(c);
                }
            }

            foreach(o; outputs)
            {
                toposort(o);
            }

            mLosses = sortedLayers
                     .map!(x => x.losses)
                     .joiner()
                     .array();
            
            mParameters = sortedLayers
                         .map!(x => x.parameters)
                         .joiner()
                         .array();

            if(mLosses.length == 0)
            {
                //This network is just for inference?
                mLoss = null;
                return;
            }

            Operation loss = mLosses[0].reshape([]);

            for(size_t i = 1; i < mLosses.length; i++)
            {
                loss = loss + mLosses[i].reshape([]);
            }

            mLoss = loss;
        }

        ///
        const(Operation)[] outputs() const
        {
            return mOutputs;
        }

        ///
        const(Operation) loss() const
        {
            return mLoss;
        }

        ///
        const(Operation)[] losses() const
        {
            return mLosses;
        }

        ///
        const(Operation)[] parameters() const
        {
            return mParameters;
        }
    }

    private
    {
        const(Operation)[] mOutputs;
        const(Operation) mLoss;
        const(Operation)[] mLosses;
        const(Operation)[] mParameters;
    }
}

/**
    Represents a layer of a neural network.

    This class encapsulates the dependencies, parameters, outputs, and auxilliary losses required for a single layer.
*/
class Layer
{
    public
    {
        ///
        this(Operation expr, const(Layer)[] deps = [], Operation[] params = [], Operation[] ls = [])
        {
            mDeps = deps;
            mParameters = params;
            mLosses = ls;
            mExpression = expr;
        }

        ///
        const(Layer)[] deps() const
        {
            return mDeps;
        }

        ///
        const(Operation)[] parameters() const
        {
            return mParameters;
        }

        ///
        const(Operation)[] losses() const
        {
            return mLosses;
        }

        ///
        const(Operation) expression() const
        {
            return mExpression;
        }
    }

    private
    {
        const(Layer)[] mDeps;
        Operation[] mParameters;
        Operation[] mLosses;
        Operation mExpression;
    }
}

