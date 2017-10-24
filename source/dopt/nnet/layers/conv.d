module dopt.nnet.layers.conv;

import dopt;
import dopt.nnet.layers.util;

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

unittest
{
    auto opts = new Conv2DOptions()
               .padding([1, 1])
               .stride([2, 2]);
    
    assert(opts.padding == [1, 1]);
    assert(opts.stride == [2, 2]);
}

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
