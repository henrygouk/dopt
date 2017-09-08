/**
    Contains an implementation of ADAM that relies on automatic differentiation

    Authors: Henry Gouk
*/
module dopt.online.adam;

import dopt.core;

/**
    Creates a delegate that can be used to perform a step using the ADAM update rule.
    
    This function relies on automatic differentiation, so the objective (which must have a volume of 1) must be
    differentiable w.r.t. all elements of wrt. The returned delegate performs minimisation.

    Params:
        objective = Operation representing the loss function to be minimised.
        wrt = an array of Operations that we want the derivative of objective with respect to.
        alpha = the step size.
        beta1 = fading factor for the first moment of the gradient.
        beta2 = fading factor for the second moment of the gradient.
        eps = to prevent division by zero.

    Returns:
         A delegate that is used to actually perform the update steps. The optimised values are stored in the "default"
         attributes of the elements of wrt.
*/
float delegate(Buffer[Operation] args) adam(Operation objective, Operation[] wrt,
    Operation alpha = float32([], [0.001f]), Operation beta1 = float32([], [0.9f]),
    Operation beta2 = float32([], [0.999f]), Operation eps = float32([], [1e-8]))
{
    import std.algorithm : map;
    import std.array : array;
    import std.range : zip;

    auto grads = grad(objective, wrt);
    auto means = wrt.map!(x => float32(x.shape)).array();
    auto vars = wrt.map!(x => float32(x.shape)).array();

    auto b1 = float32([], [1.0f]);
    auto b2 = float32([], [1.0f]);
    auto nb1 = b1 * beta1;
    auto nb2 = b2 * beta2;
    auto eta = alpha * sqrt(1.0f - nb2) / (1.0f - nb1);

    auto newMeans = grads
                   .zip(means)
                   .map!(x => beta1 * x[1] + (1.0f - beta1) * x[0])
                   .array();
    
    auto newVars = grads
                   .zip(vars)
                   .map!(x => beta2 * x[1] + (1.0f - beta2) * x[0] * x[0])
                   .array();

    auto meanHats = newMeans
                   .map!(x => x / (1.0f - beta1))
                   .array();

    auto varHats = newVars
                  .map!(x => x / (1.0f - beta2))
                  .array();

    auto newvals = zip(wrt, meanHats, varHats)
                  .map!(x => x[0] - eta * (x[1] / (sqrt(x[2]) + eps)))
                  .array();

    float update(Buffer[Operation] args)
    {
        auto newbufs = evaluate([objective] ~ newvals ~ newMeans ~ newVars ~ [nb1, nb2], args);

        foreach(b, w; zip(newbufs[1 .. $], wrt ~ means ~ vars ~ [b1, b2]))
        {
            auto wrtbuf = w.value.as!byte;
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
    auto updater = adam((yhat - y) * (yhat - y), [m, c], float32([], [1.0f]));

    //Iterate for a while
    float loss;

    for(size_t i = 0; i < 200; i++)
    {
        size_t j = i % 100;

        loss = updater([
            x: Buffer(xdata[j .. j + 1]),
            y: Buffer(ydata[j .. j + 1])
        ]);
    }

    //Print the loss after 500 iterations. Let the user decide whether it's good enough to be considered a pass.
    import std.stdio : writeln;
    writeln(
        "Adam loss: ", loss, "    ",
        "m=", m.value.as!float[0], ", ",
        "c=", c.value.as!float[0], "    ",
        "(expected m=3, c=2)");
}