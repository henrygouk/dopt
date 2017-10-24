module dopt.nnet.networks;

import std.algorithm;
import std.array;

import dopt;

class Network
{
    public
    {
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

        Operation[] inputs()
        {
            return mInputs.dup;
        }

        Operation[] outputs()
        {
            return mOutputs.dup;
        }

        Operation[] trainOutputs()
        {
            return mTrainOutputs.dup;
        }

        Operation paramLoss()
        {
            return mParameterLoss;
        }

        Projection[Operation] paramProj()
        {
            return mParameterProj;
        }

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