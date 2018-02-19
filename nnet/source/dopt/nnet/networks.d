/**
    Provides a useful tools for constructing neural networks.

    Currently only directed acyclic graphs are supported.

    Authors: Henry Gouk
*/
module dopt.nnet.networks;

import std.algorithm;
import std.array;

import dopt.core;
import dopt.nnet;
import dopt.online;

/**
    Encapsulates the details of a network with a directed acyclic graph structure.

    This class does not provide facilities to actually train the network---that can be accomplished with the 
    $(D dopt.online) package.
*/
class DAGNetwork
{
    public
    {
        /**
            Construct a DAGNetwork with the given inputs and outputs.

            Params:
                inputs = The inputs to the network. This will usually contain a single $(D Operation) representing a
                batch of feature vectors.
                outputs = The outputs (i.e., predictions) of the network.
        */
        this(Operation[] inputs, Layer[] outputs)
        {
            mInputs = inputs.dup;
            mOutputs = outputs.map!(x => x.output).array();
            mTrainOutputs = outputs.map!(x => x.trainOutput).array();

            auto layers = topologicalSort(outputs);
            auto paramsinfo = layers.map!(x => x.params).joiner().array();
            mParams = paramsinfo.map!(x => x.symbol).array();

            foreach(p; paramsinfo)
            {
                if(p.loss !is null)
                {
                    if(mParameterLoss is null)
                    {
                        mParameterLoss = p.loss;
                    }
                    else
                    {
                        mParameterLoss = mParameterLoss + p.loss;
                    }
                }

                if(p.projection !is null)
                {
                    mParameterProj[p.symbol] = p.projection;
                }
            }

            if(mParameterLoss is null)
            {
                //Prevents an annoying-to-debug segfault in user code when there are no param loss terms
                mParameterLoss = float32([], [0.0f]);
            }
        }

        /**
            The inputs provided when the $(D DAGNetwork) was constructed.
        */
        Operation[] inputs()
        {
            return mInputs.dup;
        }

        /**
            The $(D Operation) objects produced by the output layers provided during construction.
        */
        Operation[] outputs()
        {
            return mOutputs.dup;
        }

        /**
            Separate $(D Operation) objects produced by the output layers provided during constructions.

            These should be used when creating the network optimiser.
        */
        Operation[] trainOutputs()
        {
            return mTrainOutputs.dup;
        }

        /**
            The sum of all the parameter loss terms.

            This will include all the L2 weight decay terms.
        */
        Operation paramLoss()
        {
            return mParameterLoss;
        }

        /**
            An associative array of projection operations that should be applied to parameters during optimisation.
        */
        Projection[Operation] paramProj()
        {
            return mParameterProj;
        }

        /**
            An array of all the $(D Operation) nodes in the graph representing network parameters.
        */
        Operation[] params()
        {
            return mParams.dup;
        }
    }

    private
    {
        Operation[] mInputs;
        Operation[] mOutputs;
        Operation[] mTrainOutputs;
        Operation[] mParams;
        Operation mParameterLoss;
        Projection[Operation] mParameterProj;
    }
}