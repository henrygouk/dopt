/**
    Contains an implementation of the softmat activation function.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.softmax;

import dopt;

///
Layer softmax(Layer input)
{
    import dopt.core.ops.nnet : softmax;
    
    return new Layer([input], input.output.softmax(), input.trainOutput.softmax, null);
}