/**
    Contains an implementation of max pooling.
    Authors: Henry Gouk
*/
module dopt.nnet.layers.maxpool;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.online;

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