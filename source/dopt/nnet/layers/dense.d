module dopt.nnet.layers.dense;

import dopt;
import dopt.nnet.layers.util;

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