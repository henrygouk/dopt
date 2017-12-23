/**
    Authors: Henry Gouk
*/
module dopt.nnet.layers.dense;

import dopt;
import dopt.nnet.layers.util;

/**
    Encapsulates additional options for dense layers.
*/
class DenseOptions
{
    this()
    {
        _weightInit = heGaussianInit();
        _biasInit = constantInit(0.0f);
        _useBias = true;
        _weightDecay = 0;
    }

    mixin(dynamicProperties(
        "ParamInitializer", "weightInit",
        "ParamInitializer", "biasInit",
        "Projection", "weightProj",
        "Projection", "biasProj",
        "float", "weightDecay",
        "bool", "useBias"
    ));
}

///
unittest
{
    //Create a DenseOptions object with the default parameters
    auto opts = new DenseOptions()
               .weightInit(heGaussianInit())
               .biasInit(constantInit(0.0f))
               .weightProj(null)
               .biasProj(null)
               .weightDecay(0.0f)
               .useBias(true);
    
    //Options can also be read back again later
    assert(opts.weightDecay == 0.0f);
    assert(opts.useBias == true);
}

/**
    Creates a fully connected (AKA, dense) layer.

    Params:
        input = The previous layer in the network.
        numOutputs = The number of units in this layer.
        opts = Additional options with sensible default values.
    
    Returns:
        The new layer.
*/
Layer dense(Layer input, size_t numOutputs, DenseOptions opts = new DenseOptions())
{
    auto x = input.output;
    auto xTr = input.trainOutput;

    x = x.reshape([x.shape[0], x.volume / x.shape[0]]);
    xTr = xTr.reshape([xTr.shape[0], xTr.volume / xTr.shape[0]]);

    auto weights = float32([numOutputs, x.shape[1]]);
    opts._weightInit(weights);

    auto weightLoss = (opts.weightDecay == 0.0f) ? null : (opts.weightDecay * sum(weights * weights));

    Parameter[] params = [
        Parameter(weights, weightLoss, opts.weightProj)
    ];

    auto y = matmul(x, weights.transpose([1, 0]));
    auto yTr = matmul(xTr, weights.transpose([1, 0]));

    if(opts.useBias)
    {
        auto bias = float32([numOutputs]);
        opts._biasInit(bias);

        y = y + bias.repeat(y.shape[0]);
        yTr = yTr + bias.repeat(yTr.shape[0]);

        params ~= Parameter(bias, null, opts.biasProj);
    }

    return new Layer([input], y, yTr, params);
}