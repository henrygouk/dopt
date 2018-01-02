/**
    Contains generic utilities for working with $(D Layer) objects.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers;

import dopt;

public
{
    import dopt.nnet.layers.batchnorm;
    import dopt.nnet.layers.conv;
    import dopt.nnet.layers.datasource;
    import dopt.nnet.layers.dense;
    import dopt.nnet.layers.dropout;
    import dopt.nnet.layers.maxpool;
    import dopt.nnet.layers.relu;
    import dopt.nnet.layers.softmax;
}

/**
    Encapsulates the expressions and parameter information that defines a network layer.
*/
class Layer
{
    public
    {
        /**
            Constructs a new layer.

            Params:
                deps = Other $(D Layer) objects that this layer depends on.
                outExpr = The output expression to use at test time.
                trainOutExpr = The output expression to use at train time.
                params = Any parameters managed by this layer.
        */
        this(Layer[] deps, Operation outExpr, Operation trainOutExpr, Parameter[] params)
        {
            mDeps = deps.dup;
            mParams = params.dup;
            mOutput = outExpr;
            mTrainOutput = trainOutExpr;
        }

        Layer[] deps()
        {
            return mDeps.dup;
        }

        Parameter[] params()
        {
            return mParams.dup;
        }

        Operation output()
        {
            return mOutput;
        }

        Operation trainOutput()
        {
            return mTrainOutput;
        }
    }

    private
    {
        Layer[] mDeps;
        Parameter[] mParams;
        Operation mOutput;
        Operation mTrainOutput;
    }
}

Layer[] topologicalSort(Layer[] ops)
{
    Layer[] sortedOps;

    void toposort(Layer o)
    {
        import std.algorithm : canFind;

        if(sortedOps.canFind(o))
        {
            return;
        }

        foreach(d; o.deps)
        {
            toposort(d);
        }
        
        sortedOps ~= o;
    }

    foreach(o; ops)
    {
        toposort(o);
    }

    return sortedOps;
}