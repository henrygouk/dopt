/**
    Contains an implementation of the softmat activation function.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.softmax;

import dopt.core;
import dopt.nnet;
import dopt.nnet.layers.util;
import dopt.online;

///
Layer softmax(Layer input)
{
    import dopt.core.ops.nnet : softmax;
    
    return new Layer([input], input.output.softmax(), input.trainOutput.softmax, null);
}