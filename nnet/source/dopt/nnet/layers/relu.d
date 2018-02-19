/**
    Contains an implementation of the ReLU activation function.
    
    Authors: Henry Gouk
*/
module dopt.nnet.layers.relu;

import dopt.core;
import dopt.nnet;
import dopt.nnet.layers.util;
import dopt.online;

///
Layer relu(Layer input)
{
    Operation reluImpl(Operation x)
    {
        import dopt.core.ops.nnet : relu;
        
        return relu(x);
    }

    return new Layer([input], reluImpl(input.output), reluImpl(input.trainOutput), null);
}