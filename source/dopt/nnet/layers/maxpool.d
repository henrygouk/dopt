/**
    Contains an implementation of max pooling.
    Authors: Henry Gouk
*/
module dopt.nnet.layers.maxpool;

import dopt;
import dopt.nnet.layers.util;

///
Layer maxPool(Layer input, size_t[] dims)
{
    import dopt.core.ops.nnet : maxpool;
    
    return new Layer(
        [input],
        input.output.maxpool(dims),
        input.trainOutput.maxpool(dims),
        null
    );
}