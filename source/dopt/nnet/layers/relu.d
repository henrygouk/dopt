module dopt.nnet.layers.relu;

import dopt;
import dopt.nnet.layers.util;

Layer relu(Layer input)
{
    Operation reluImpl(Operation x)
    {
        import std.array : array;
        import std.range : repeat;

        auto zeros = float32(x.shape, repeat(0.0f, x.volume).array());
        auto cmp = x.gt(zeros);

        return cmp * x;
    }

    return new Layer([input], reluImpl(input.output), reluImpl(input.trainOutput), null);
}