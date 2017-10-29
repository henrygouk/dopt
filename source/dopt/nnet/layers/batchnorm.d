/**
    Authors: Henry Gouk
*/
module dopt.nnet.layers.batchnorm;

import dopt;

///
Layer batchNorm(Layer input)
{
    import std.array : array;
    import std.range : repeat;

    auto x = input.output;
    auto xTr = input.trainOutput;

    auto gamma = float32([1, x.shape[1], 1, 1], repeat(1.0f, x.shape[1]).array());
    auto beta = float32([x.shape[1]]);

    auto y = x.batchNormTrain(gamma, beta);
    auto yTr = xTr.batchNormTrain(gamma, beta);

    return new Layer([input], y, yTr, [Parameter(gamma, null, null), Parameter(beta, null, null)]);
}