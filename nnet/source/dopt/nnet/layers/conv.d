/**
    Contains an implementation of convolutional layers.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.conv;

import dopt.core;
import dopt.nnet;
import dopt.nnet.layers.util;
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

    }

    mixin(dynamicProperties(
        "size_t[]", "padding",
        "size_t[]", "stride",
        "ParamInitializer", "filterInit",
        "ParamInitializer", "biasInit",
        "Projection", "filterProj",
        "Projection", "biasProj",
        "float", "weightDecay",
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

    auto x = input.output;
    auto xTr = input.trainOutput;

    auto filters = float32([outputChannels, x.shape[1]] ~ filterDims);
    filterInit(filters);

    import std.math : isNaN;

    auto filterLoss = (weightDecay == 0.0f) ? null : (weightDecay * sum(filters * filters));

    Parameter[] params = [
            Parameter(filters, filterLoss, filterProj)
        ];

    auto y = x.convolution(filters, padding, stride);
    auto yTr = xTr.convolution(filters, padding, stride);

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
