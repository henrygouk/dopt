/**
    Contains an implementation of batch normalisation.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.batchnorm;

import dopt.core;
import dopt.nnet;
import dopt.nnet.layers.util;
import dopt.online;

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
        _momentum = 0.9f;
        _maxgain = float.infinity;
    }

    mixin(dynamicProperties(
        "ParamInitializer", "gammaInit",
        "ParamInitializer", "betaInit",
        "Projection", "gammaProj",
        "Projection", "betaProj",
        "float", "maxgain",
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
               .betaProj(null)
               .gammaDecay(0.0f)
               .momentum(0.9f);
    
    //Options can also be read back again later
    assert(opts.gammaDecay == 0.0f);
    assert(opts.momentum == 0.9f);
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

    auto before = xTr;
    auto zeros = float32Constant([before.shape[1]], repeat(0.0f, before.shape[1]).array());
    auto after = before.batchNormInference(gamma, zeros, zeros, var);

    before = before.reshape([before.shape[0], before.volume / before.shape[0]]);
    after = after.reshape([after.shape[0], after.volume / after.shape[0]]);

    Operation maxGainProj(Operation newGamma)
    {
        auto beforeNorms = sum(before * before, [1]) + 1e-8;
        auto afterNorms = sum(after * after, [1]) + 1e-8;
        auto mg = maxElement(sqrt(afterNorms / beforeNorms));

        if(opts._gammaProj is null)
        {
            return newGamma * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain));
        }
        else
        {
            return opts._gammaProj(newGamma * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain)));
        }
    }

    Projection gammaProj = opts._gammaProj;

    if(opts.maxgain != float.infinity)
    {
        gammaProj = &maxGainProj;
    }

    Operation meanUpdater(Operation ignored)
    {
        return meanUpdateSym;
    }

    Operation varUpdater(Operation ignored)
    {
        return varUpdateSym;
    }

    return new Layer([input], y, yTr, [
        Parameter(gamma, opts._gammaDecay == 0.0f ? null : opts._gammaDecay * sum(gamma * gamma), gammaProj),
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
    
    assert(approxEqual(layers.params[2].symbol.value.get!float, [3.0f, 4.0f]));
}