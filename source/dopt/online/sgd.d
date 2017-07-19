module dopt.online.sgd;

import dopt.core;

/**
    Creates a function that can be used to perform a step using the stochastic gradient descent update rule. This
    function relies on automatic differentiation, so the objective (which must have a volume of 1) must be
    differentiable w.r.t. all elements of wrt. This function performs minimisation.

    objective is the loss function to be minimised
    wrt are the parameters of the functions that this function is allowed to change
    learningRate is the value used to scale the size of the gradient used in the update rule

    This function returns a delegate that is used to actually perform the update steps. The optimised values are
    stored in the "default" attributes of the elements of wrt.
*/
Buffer delegate(Buffer[const(Operation)] args) sgd(const(Operation) objective, Operation[] wrt,
    const(Operation) learningRate = float32([], [0.1f]))
{
    import std.algorithm : map;
    import std.array : array;
    import std.range : zip;

    auto grads = grad(objective, wrt);

    auto newvals = zip(wrt, grads)
                  .map!(x => x[0] - learningRate * x[1])
                  .array();

    Buffer update(Buffer[const(Operation)] args)
    {
        auto newbufs = evaluate([objective] ~ newvals, args);

        foreach(b, w; zip(newbufs[1 .. $], wrt))
        {
            auto wrtbuf = cast(byte[])w.attributes["default"].get!Buffer.as!byte;
            wrtbuf[] = b.as!byte[];
        }

        return newbufs[0];
    }

    return &update;
}