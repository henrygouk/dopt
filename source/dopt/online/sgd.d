/**
    Contains an implementation of stochastic gradient descent that relies on automatic differentiation

    Authors: Henry Gouk
*/
module dopt.online.sgd;

import dopt.core;

/**
    Creates a delegate that can be used to perform a step using the stochastic gradient descent update rule.
    
    This function relies on automatic differentiation, so the objective (which must have a volume of 1) must be
    differentiable w.r.t. all elements of wrt. The returned delegate performs minimisation.

    Params:
        objective = Operation representing the loss function to be minimised.
        wrt = an array of Operations that we want the derivative of objective with respect to.
        learningRate = the value used to scale the size of the gradient used in the update rule

    Returns:
         A delegate that is used to actually perform the update steps. The optimised values are stored in the "default"
         attributes of the elements of wrt.
*/
float delegate(Buffer[const(Operation)] args) sgd(const(Operation) objective, Operation[] wrt,
    const(Operation) learningRate = float32([], [0.01f]))
{
    import std.algorithm : map;
    import std.array : array;
    import std.range : zip;

    auto grads = grad(objective, wrt);

    auto newvals = zip(wrt, grads)
                  .map!(x => x[0] - learningRate * x[1])
                  .array();

    float update(Buffer[const(Operation)] args)
    {
        auto newbufs = evaluate([objective] ~ newvals, args);

        foreach(b, w; zip(newbufs[1 .. $], wrt))
        {
            auto wrtbuf = cast(byte[])w.attributes["default"].get!Buffer.as!byte;
            wrtbuf[] = b.as!byte[];
        }

        return newbufs[0].as!float[0];
    }

    return &update;
}

///
unittest
{
    import std.random : uniform;

    //Generate some points
    auto xdata = new float[100];
    auto ydata = new float[100];

    foreach(i; 0 .. 100)
    {
        xdata[i] = uniform(-10.0f, 10.0f);
        ydata[i] = 3.0f * xdata[i] + 2.0f;
    }

    //Create the model
    auto x = float32([]);
    auto m = float32([]);
    auto c = float32([]);

    auto yhat = m * x + c;
    auto y = float32([]);

    //Create an SGD updater
    auto updater = sgd((yhat - y) * (yhat - y), [m, c]);

    //Iterate for a while
    float loss;

    for(size_t i = 0; i < 500; i++)
    {
        size_t j = i % 100;

        loss = updater([
            x: Buffer(xdata[j .. j + 1]),
            y: Buffer(ydata[j .. j + 1])
        ]);
    }

    //Print the loss after 500 iterations. Let the user decide whether it's good enough to be considered a pass.
    import std.stdio : writeln;
    writeln("SGD loss: ", loss);
}