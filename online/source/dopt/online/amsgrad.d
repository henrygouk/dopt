/**
    Contains an implementation of AMSGrad that relies on automatic differentiation

    Authors: Henry Gouk
*/
module dopt.online.amsgrad;

import dopt.core;
import dopt.online;

/**
    Creates a delegate that can be used to perform a step using the AMSGrad update rule.
    
    This function relies on automatic differentiation, so the objective (which must have a volume of 1) must be
    differentiable w.r.t. all elements of wrt. The returned delegate performs minimisation.

    Params:
        outputs = An array of outputs. The first element of this array is the objective function to be minimised.
        wrt = An array of Operations that we want the derivative of objective with respect to.
        projs = Projection functions that can be applied when updating the values of elements in $(D wrt).
        alpha = The step size.
        beta1 = Fading factor for the first moment of the gradient.
        beta2 = Fading factor for the second moment of the gradient.
        eps = To prevent division by zero.

    Returns:
         A delegate that is used to actually perform the update steps. The optimised values are stored in the
         $(D value) properties of the elements of $(D wrt). The delegate returns the values computed for each element of the
         $(D outputs) array. This can be useful for keeping track of several different performance metrics in a
         prequential manner.
*/
Updater amsgrad(Operation[] outputs, Operation[] wrt, Projection[Operation] projs,
    Operation alpha = float32([], [0.001f]), Operation beta1 = float32([], [0.9f]),
    Operation beta2 = float32([], [0.999f]), Operation eps = float32([], [1e-8]))
{
    import std.algorithm : map;
    import std.array : array;
    import std.range : zip;

    auto objective = outputs[0];

    auto grads = grad(objective, wrt);
    auto means = wrt.map!(x => float32(x.shape)).array();
    auto vars = wrt.map!(x => float32(x.shape)).array();
    auto varhats = wrt.map!(x => float32(x.shape)).array();

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

    auto newVarhats = varhats
                     .zip(vars)
                     .map!(x => max(x[0], x[1]))
                     .array();

    auto newvals = zip(wrt, newMeans, newVars)
                  .map!(x => x[0] - eta * (x[1] / (sqrt(x[2]) + eps)))
                  .array();

    //Apply projections
    for(size_t i = 0; i < newvals.length; i++)
    {
        if(wrt[i] in projs)
        {
            newvals[i] = projs[wrt[i]](newvals[i]);
        }
    }

    auto updatePlan = compile(outputs ~ newvals ~ newMeans ~ newVars ~ newVarhats ~ [nb1, nb2]);

    import std.range : chain;

    auto newbufs = chain(wrt, means, vars, varhats, [b1, b2])
                  .map!(x => x.value)
                  .array();

    newbufs = outputs.map!(x => allocate(x.volume * x.elementType.sizeOf)).array() ~ newbufs;

    DeviceBuffer[] update(DeviceBuffer[Operation] args)
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

    //Create an AMSGrad updater
    auto updater = amsgrad([(yhat - y) * (yhat - y)], [m, c], null, float32([], [0.1f]));

    //Iterate for a while
    float loss;

    for(size_t i = 0; i < 300; i++)
    {
        size_t j = i % 100;

        loss = updater([
            x: buffer(xdata[j .. j + 1]),
            y: buffer(ydata[j .. j + 1])
        ])[0].get!float[0];
    }

    //Print the loss after 200 iterations. Let the user decide whether it's good enough to be considered a pass.
    import std.stdio : writeln;
    writeln(
        "AMSGrad loss: ", loss, "    ",
        "m=", m.value.get!float[0], ", ",
        "c=", c.value.get!float[0], "    ",
        "(expected m=3, c=2)");
}