module dopt.nnet.layers;

import dopt;

public
{
    import dopt.nnet.layers.conv;
    import dopt.nnet.layers.datasource;
    import dopt.nnet.layers.dense;
    import dopt.nnet.layers.dropout;
    import dopt.nnet.layers.maxpool;
    import dopt.nnet.layers.relu;
    import dopt.nnet.layers.softmax;
}

class Layer
{
    public
    {
        this(Layer[] deps, Operation outExpr, Operation trainOutExpr, Parameter[] params)
        {
            mDeps = deps.dup;
            mParams = params.dup;
            mOutput = outExpr;
            mTrainOutput = trainOutExpr;

            import std.exception : enforce;

            enforce(mOutput.shape == mTrainOutput.shape,
                "The shapes of the output and trainOutput expressions must be the same");
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