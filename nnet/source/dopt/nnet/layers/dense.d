/**
    Contains an implementation of dense (i.e., fully connected) layers.
    Authors: Henry Gouk
*/
module dopt.nnet.layers.dense;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.online;

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
        _maxgain = float.infinity;
        _spectralDecay = 0.0f;
    }

    mixin(dynamicProperties(
        "ParamInitializer", "weightInit",
        "ParamInitializer", "biasInit",
        "Projection", "weightProj",
        "Projection", "biasProj",
        "float", "maxgain",
        "float", "weightDecay",
        "float", "spectralDecay",
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
    Operation safeAdd(Operation op1, Operation op2)
    {
        if(op1 is null && op2 is null)
        {
            return null;
        }
        else if(op1 is null)
        {
            return op2;
        }
        else if(op2 is null)
        {
            return op1;
        }
        else
        {
            return op1 + op2;
        }
    }

    auto x = input.output;
    auto xTr = input.trainOutput;

    x = x.reshape([x.shape[0], x.volume / x.shape[0]]);
    xTr = xTr.reshape([xTr.shape[0], xTr.volume / xTr.shape[0]]);

    auto weights = float32([numOutputs, x.shape[1]]);
    opts._weightInit(weights);

    Operation weightLoss;
    weightLoss = safeAdd(weightLoss, (opts.weightDecay == 0.0f) ? null : (opts.weightDecay * sum(weights * weights)));
    weightLoss = safeAdd(
        weightLoss,
        (opts.spectralDecay == 0.0f) ? null : (opts.spectralDecay * spectralNorm(weights))
    );

    auto weightProj = opts._weightProj;

    auto y = matmul(x, weights.transpose([1, 0]));
    auto yTr = matmul(xTr, weights.transpose([1, 0]));

    auto before = xTr.reshape([xTr.shape[0], xTr.volume / xTr.shape[0]]);
    auto after = yTr.reshape([yTr.shape[0], yTr.volume / yTr.shape[0]]);

    Operation maxGainProj(Operation newWeights)
    {
        auto beforeNorms = sum(before * before, [1]) + 1e-8;
        auto afterNorms = sum(after * after, [1]) + 1e-8;
        auto mg = maxElement(sqrt(afterNorms / beforeNorms));

        if(opts.weightProj is null)
        {
            return newWeights * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain));
        }
        else
        {
            return opts._weightProj(newWeights * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain)));
        }
    }

    if(opts.maxgain != float.infinity)
    {
        weightProj = &maxGainProj;
    }

    Parameter[] params = [
        Parameter(weights, weightLoss, weightProj)
    ];

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

private Operation spectralNorm(Operation weights, size_t numIts = 1)
{
    auto x = uniformSample([weights.shape[0], 1]) * 2.0f - 1.0f;

    for(int i = 0; i < numIts; i++)
    {
        x = matmul(weights.transpose([1, 0]), matmul(weights, x));
    }

    auto v = x / sqrt(sum(x * x));
    auto y = matmul(weights, v);

    return sum(y * y);
}