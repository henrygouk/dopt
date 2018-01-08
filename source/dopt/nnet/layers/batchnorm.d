/**
    Contains an implementation of batch normalisation.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.batchnorm;

import dopt;

import dopt.nnet.layers.util;

/**
    Encapsulates additional options for batchnorm layers.
*/
class BatchNormOptions
{
    this()
    {
        _gammaInit = constantInit(1.0f);
        _betaInit = constantInit(0.0f);
        _gammaDecay = 0;
        _momentum = 0.99f;
    }

    mixin(dynamicProperties(
        "ParamInitializer", "gammaInit",
        "ParamInitializer", "betaInit",
        "Projection", "gammaProj",
        "Projection", "betaProj",
        "float", "gammaDecay",
        "float", "momentum"
    ));
}

///
unittest
{
    //Create a BatchNormOptions object with the default parameters
    auto opts = new BatchNormOptions()
               .gammaInit(constantInit(1.0f))
               .betaInit(constantInit(0.0f))
               .gammaProj(null)
               .gammaProj(null)
               .gammaDecay(0.0f)
               .momentum(0.99f);
    
    //Options can also be read back again later
    assert(opts.gammaDecay == 0.0f);
    assert(opts.momentum == 0.99f);
}

///
Layer batchNorm(Layer input, BatchNormOptions opts = new BatchNormOptions())
{
    /*Appologies to anyone trying to understand how I've implemented BN---this is a bit hacky!
      What we're doing is packing the running mean/variance estimate provided during the training
      forward propagation into the same tensor as the normalised layer activations. The batchNormTrain
      function then seperates these out into 3 different operation nodes. We can then use the projected
      gradient descent operator to constrain the mean/var model parameters to be equal to these running
      statistics.
    */

    import std.array : array;
    import std.range : repeat;

    auto x = input.output;
    auto xTr = input.trainOutput;

    auto gamma = float32([1, x.shape[1], 1, 1]);
    auto beta = float32([x.shape[1]]);

    opts._gammaInit(gamma);
    opts._betaInit(beta);

    auto mean = float32([x.shape[1]]);
    auto var = float32([x.shape[1]], repeat(1.0f, x.shape[1]).array());

    auto bnop = xTr.batchNormTrain(gamma, beta, mean, var, opts._momentum);
    auto yTr = bnop[0];
    auto meanUpdateSym = bnop[1];
    auto varUpdateSym = bnop[2];

    auto y = x.batchNormInference(gamma, beta, mean, var);

    Operation meanUpdater(Operation ignored)
    {
        return meanUpdateSym;
    }

    Operation varUpdater(Operation ignored)
    {
        return varUpdateSym;
    }

    return new Layer([input], y, yTr, [
        Parameter(gamma, opts._gammaDecay == 0.0f ? null : opts._gammaDecay * sum(gamma * gamma), opts._gammaProj),
        Parameter(beta, null, opts._betaProj),
        Parameter(mean, null, &meanUpdater),
        Parameter(var, null, &varUpdater)
    ]);
}

unittest
{
    auto x = float32([3, 2], [1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f]);
    
    auto layers = dataSource(x).batchNorm();
    auto network = new DAGNetwork([x], [layers]);

    auto trloss = layers.trainOutput.sum();

    auto updater = adam([trloss], network.params, network.paramProj);

    for(size_t i = 0; i < 1000; i++)
    {
        updater(null);
    }

    import std.math : approxEqual;
    
    assert(approxEqual(layers.params[2].symbol.value.as!float, [3.0f, 4.0f]));
}