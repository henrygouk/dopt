/**
    Contains an implementation of stochastic gradient descent that relies on automatic differentiation

    Authors: Henry Gouk
*/
module dopt.online.sgd;

import dopt.core;
import dopt.online;

/**
    Creates a delegate that can be used to perform a step using the stochastic gradient descent update rule.
    
    This function relies on automatic differentiation, so the objective (which must have a volume of 1) must be
    differentiable w.r.t. all elements of wrt. The returned delegate performs minimisation.

    Params:
        objective = Operation representing the loss function to be minimised.
        wrt = an array of Operations that we want the derivative of objective with respect to.
        learningRate = the value used to scale the size of the gradient used in the update rule
        momentumRate = scaling factor for the previous update

    Returns:
         A delegate that is used to actually perform the update steps. The optimised values are stored in the "default"
         attributes of the elements of wrt.
*/
Updater sgd(Operation[] outputs, Operation[] wrt,
    Operation learningRate = float32([], [0.01f]), Operation momentumRate = float32([], [0.0f]))
{
    import std.algorithm : map;
    import std.array : array;
    import std.range : zip;

    auto objective = outputs[0];

    auto grads = grad(objective, wrt);

    auto momentum = grads
                   .map!(x => float32(x.shape))
                   .array();
    
    auto newMomentum = zip(grads, momentum)
                      .map!(x => x[1] * momentumRate + learningRate * x[0])
                      .array();

    auto newvals = zip(wrt, newMomentum)
                  .map!(x => x[0] - x[1])
                  .array();

    auto updatePlan = compile(outputs ~ newvals ~ newMomentum);

    import std.range : chain;

    auto newbufs = chain(wrt, momentum)
                  .map!(x => x.value)
                  .array();

    newbufs = outputs.map!(x => Buffer(new ubyte[x.volume * x.elementType.sizeOf])).array() ~ newbufs;

    Buffer[] update(Buffer[Operation] args)
    {
        updatePlan.execute(args, newbufs);

        return newbufs[0 .. outputs.length];
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
    auto updater = sgd([(yhat - y) * (yhat - y)], [m, c], float32([], [0.001f]), float32([], [0.9f]));

    //Iterate for a while
    float loss;

    for(size_t i = 0; i < 300; i++)
    {
        size_t j = i % 100;

        loss = updater([
            x: Buffer(xdata[j .. j + 1]),
            y: Buffer(ydata[j .. j + 1])
        ])[0].as!float[0];
    }

    //Print the loss after 500 iterations. Let the user decide whether it's good enough to be considered a pass.
    import std.stdio : writeln;
    writeln(
        "SGD loss: ", loss, "    ",
        "m=", m.value.as!float[0], ", ",
        "c=", c.value.as!float[0], "    ",
        "(expected m=3, c=2)");
}