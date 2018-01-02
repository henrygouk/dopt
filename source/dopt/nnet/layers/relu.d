/**
    Contains an implementation of the ReLU activation function.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.relu;

import dopt;
import dopt.nnet.layers.util;

///
Layer relu(Layer input)
{
    Operation reluImpl(Operation x)
    {
        import std.array : array;
        import std.range : repeat;

        auto zeros = float32Constant(x.shape, repeat(0.0f, x.volume).array());
        
        return max(x, zeros);
    }

    return new Layer([input], reluImpl(input.output), reluImpl(input.trainOutput), null);
}