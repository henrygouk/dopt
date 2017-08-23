/**
    Contains constructors for common neural network layers.

    Authors: Henry Gouk
*/
module dopt.nnet.layers;

import dopt.core;
import dopt.nnet;

/**
    Creates a neural network layer that simply outputs the value of a variable.

    Params:
        variable = The variable that dictates the output value of this layer.

    Returns:
        A $(D Layer).
*/
Layer dataSource(Operation variable)
{
    return new Layer(variable);
}

/**
    Creates a fully connected layer.

    If $(D biasInit) is set to null, then no biases are added to the output.

    Params:
        inputs = The input to the new layer.
        units = The number of fully connected units in the new layer.
        weightInit = A $(D ParamInitializer) used to initialise the weight matrix.
        biasInit = A $(D ParamInitializer) used to initialize the bias vector.
    
    Returns:
        A new $(D Layer) representing a linear transform.
*/
Layer dense(const(Layer) inputs, size_t units, ParamInitializer weightInit = glorotGaussianInit(),
    ParamInitializer biasInit = constantInit(0.0f))
{
    //Flatten the input features, if required
    auto tmp = inputs.expression;
    auto x = tmp.rank > 2 ? tmp.reshape([tmp.shape[0], tmp.volume / tmp.shape[0]]) : tmp;

    auto reshapedInput = x.reshape([x.shape[0], x.volume / x.shape[0]]);
    auto weights = float32([reshapedInput.shape[1], units]);
    weightInit(weights);

    auto expr = matmul(reshapedInput, weights);

    auto params = [weights];

    if(biasInit !is null)
    {
        auto biases = float32([units]);
        biasInit(biases);
        params ~= biases;
        expr = expr + biases.repeat(x.shape[0]);
    }

    return new Layer(expr, [inputs], params);
}

/**
    Creates a layer representing a Rectified Linear Unit activation function.

    Params:
        inputs = The input to the new layer.
    
    Returns:
        The new ReLU $(D Layer).
*/
Layer relu(const(Layer) inputs)
{
    auto x = inputs.expression;
    auto zeros = float32([], [0.0f]).repeat(x.volume).reshape(x.shape);

    return new Layer(max(x, zeros), [inputs]);
}

/**
    Creates a layer representing a softmax activation function.

    Params:
        inputs = The input to the new layer.
    
    Returns:
        The new softmax $(D Layer).
*/
Layer softmax(const(Layer) inputs)
{
    import dopt.core.ops.nnet : softmax;

    auto x = inputs.expression;

    return new Layer(softmax(x), [inputs]);
}

/**
    Creates a cross entropy loss layer.

    Params:
        inputs = The predicted class label distribution.
        labels = The true class label distribution.
    
    Returns:
        The new cross entropy $(D Layer).
*/
Layer crossEntropy(const(Layer) inputs, const(Layer) labels)
{
    auto x = inputs.expression;
    auto y = labels.expression;
    
    auto expr = sum(y * log(x)) * (-1.0f / x.shape[0]);

    return new Layer(expr, [inputs], [], [expr]);
}

/**
    Creates a convolutional layer.

    Params:
        inputs = An input layer of feature maps.
        numMaps = The number of feature maps that should be produced by the new layer.
        filterDims = The dimensions of the filters used in the new layer.
        filterInit = A ParamInitializer used to initialise the filter parameters of the new layer.
        biasInit = A ParamInitializer used to initialise the bias vector for the new layer.
    
    Returns:
        A new convolutional $(D Layer).
*/
Layer convolutional(const(Layer) inputs, size_t numMaps, const(size_t)[] filterDims,
    ParamInitializer filterInit = glorotGaussianInit(), ParamInitializer biasInit = constantInit(0.0f))
{
    auto x = inputs.expression;

    auto filters = float32([numMaps, x.shape[1]] ~ filterDims);
    filterInit(filters);

    auto b = float32([1, numMaps, 1, 1]);
    biasInit(b);

    auto z = convolution(x, filters);
    z = z + b.repeat([x.shape[0], 1, z.shape[2], z.shape[3]]);

    return new Layer(z, [inputs], [filters, b]);
}

/**
    Creates a max pooling layer.

    Params:
        inputs = The input layer.
        poolDims = The dimensions of the pooling regions.
    
    Returns:
        A new max pooling $(D Layer).
*/
Layer maxPool(const(Layer) inputs, const(size_t)[] poolDims)
{
    return new Layer(inputs.expression.maxpool(poolDims), [inputs]);
}
