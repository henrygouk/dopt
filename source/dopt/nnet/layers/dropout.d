/**
    Authors: Henry Gouk
*/
module dopt.nnet.layers.dropout;

import dopt;

Layer dropout(Layer input, float dropProb)
{
    import std.array : array;
    import std.range : repeat;

    auto x = input.output;
    auto xTr = input.trainOutput;
    
    auto dropMask = float32(xTr.shape, repeat(dropProb, xTr.volume).array());
    auto yTr = uniformSample(xTr.shape).gt(dropMask) * xTr;

    auto scale = float32(x.shape, repeat((1.0f - dropProb), x.volume).array());
    auto y = x * scale;

    return new Layer([input], y, yTr, null);
}