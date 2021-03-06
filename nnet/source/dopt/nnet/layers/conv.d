/**
    Contains an implementation of convolutional layers.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.conv;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.online;

/**
    Encapsulates the additional options for a $(D Layer) created with conv2D.
*/
class Conv2DOptions
{
    this()
    {
        _useBias = true;
        _filterInit = heGaussianInit();
        _biasInit = constantInit(0.0f);
        _padding = [0, 0];
        _stride = [1, 1];
        _weightDecay = 0.0f;
        _maxgain = float.infinity;
        _spectralDecay = 0.0f;

    }

    mixin(dynamicProperties(
        "size_t[]", "padding",
        "size_t[]", "stride",
        "ParamInitializer", "filterInit",
        "ParamInitializer", "biasInit",
        "Projection", "filterProj",
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
    //Creates a Conv2DOptions object with the default parameter values
    auto opts = new Conv2DOptions()
               .padding([0, 0])
               .stride([1, 1])
               .filterInit(heGaussianInit())
               .biasInit(constantInit(0.0f))
               .filterProj(null)
               .biasProj(null)
               .weightDecay(0.0f)
               .useBias(true);
    
    //The fields can also be accessed again later
    assert(opts.padding == [0, 0]);
    assert(opts.stride == [1, 1]);
}

/**
    Creates a convolutional layer typically found in a convnet used for image classification.

    Params:
        input = The previous (i.e., input) layer.
        outputChannels = The number of feature maps that this layer should produce.
        filterDims = The size of the kernels that should be convolved with the inputs.
        opts = Additional options, with sensible defaults.
    
    Returns:
        The new convolutional $(D Layer).
*/
Layer conv2D(Layer input, size_t outputChannels, size_t[] filterDims, Conv2DOptions opts = new Conv2DOptions())
{
    auto padding = opts.padding;
    auto stride = opts.stride;
    auto filterInit = opts.filterInit;
    auto biasInit = opts.biasInit;
    auto filterProj = opts.filterProj;
    auto biasProj = opts.biasProj;
    auto weightDecay = opts.weightDecay;
    auto useBias = opts.useBias;
    auto spectralDecay = opts.spectralDecay;

    auto x = input.output;
    auto xTr = input.trainOutput;

    auto filters = float32([outputChannels, x.shape[1]] ~ filterDims);
    filterInit(filters);

    import std.math : isNaN;

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

    Operation filterLoss;
    filterLoss = safeAdd(filterLoss, (weightDecay == 0.0f) ? null : (weightDecay * sum(filters * filters)));
    filterLoss = safeAdd(
        filterLoss,
        (spectralDecay == 0.0f) ? null : spectralDecay * spectralNorm(filters, padding, stride)
    );

    auto y = x.convolution(filters, padding, stride);
    auto yTr = xTr.convolution(filters, padding, stride);

    auto before = xTr.reshape([xTr.shape[0], xTr.volume / xTr.shape[0]]);
    auto after = yTr.reshape([yTr.shape[0], yTr.volume / yTr.shape[0]]);

    Operation maxGainProj(Operation newWeights)
    {
        auto beforeNorms = sum(before * before, [1]) + 1e-8;
        auto afterNorms = sum(after * after, [1]) + 1e-8;
        auto mg = maxElement(sqrt(afterNorms / beforeNorms));

        if(opts.filterProj is null)
        {
            return newWeights * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain));
        }
        else
        {
            return opts._filterProj(newWeights * (1.0f / max(float32Constant([], [1.0f]), mg / opts.maxgain)));
        }
    }

    if(opts.maxgain != float.infinity)
    {
        filterProj = &maxGainProj;
    }

    Parameter[] params = [
            Parameter(filters, filterLoss, filterProj)
        ];

    if(useBias)
    {
        auto biases = float32([outputChannels]);
        biasInit(biases);

        y = y.addBias(biases);
        yTr = yTr.addBias(biases);

        params ~= Parameter(biases, null, biasProj);
    }

    return new Layer([input], y, yTr, params);
}

/**
    Note that this function computes the incorrect norm used by Yoshida and Miyato (2017).

    Yoshida, Y., & Miyato, T. (2017). Spectral Norm Regularization for Improving the Generalizability of Deep Learning.
    arXiv preprint arXiv:1705.10941.
*/
private Operation spectralNorm(Operation filters, size_t[] padding, size_t[] stride, size_t numIts = 1)
{
    filters = filters.reshape([filters.shape[0], filters.volume / filters.shape[0]]);
    auto x = uniformSample([filters.shape[1], 1]) * 2.0f - 1.0f;

    for(int i = 0; i < numIts; i++)
    {
        x = matmul(filters.transpose([1, 0]), matmul(filters, x));
    }

    auto v = x / sqrt(sum(x * x));
    auto y = matmul(filters, v);

    return sum(y * y);
}