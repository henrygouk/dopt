/**
    Contains the automatic differentiation framework.

    Authors: Henry Gouk
*/
module dopt.core.grads;

import std.exception;

import dopt.core.grads.basic;
import dopt.core.grads.math;
import dopt.core.grads.nnet;
import dopt.core.ops;
import dopt.core.types;

alias Gradient = Operation[] delegate(Operation op, Operation parentGrad);

void initialize()
{
    dopt.core.grads.basic.initialize();
    dopt.core.grads.math.initialize();
    dopt.core.grads.nnet.initialize();
}

/**
    Computes the gradient of a scalar-valued operation with respect to several dependencies.

    This function provides an implementation of automatic differentiation can be used for greatly simplifying the
    process of optimising objective functions. The particular technique used by the function is known as
    reverse mode automatic differentiation.

    Params:
        objective = The function being differentiated.
        wrt = The (indirect) dependencies that $(D objective) is being differentiated with respect to.

    Returns:
        An array of operations that evaluate to the derivative of $(D objective) to each of the elements of $(D wrt).
*/
Operation[] grad(Operation objective, Operation[] wrt)
{
    import std.algorithm : canFind, countUntil, map;
    import std.array : array;
    import std.conv : to;
    import std.range : retro, zip;

    enforce(objective.outputType.volume == 1, "The objective must have a volume of one");
    enforce(objective.outputType.elementType == DataType.float32, "The objective must have a floating point type");

    Operation[] ops;

    void traverse(Operation op)
    {
        foreach(d; op.deps)
        {
            if(!ops.canFind(d))
            {
                traverse(d);
            }
        }

        ops ~= op;
    }

    //Topologically sort the operations
    traverse(objective);

    Operation[Operation] grads;

    //TODO: when I implement a 'ones' operation, replace this line
    grads[objective] = float32(objective.outputType.shape, [1.0f]);

    //Iterate through the operations in reverse order (reverse mode autodiff)
    foreach(op; ops.retro)
    {
        //Get the function that will let us compute the gradient of op w.r.t. its deps
        auto gradFunc = mGradients.get(op.opType, null);
        auto opGrad = grads.get(op, null);
        
        if(gradFunc is null || opGrad is null)
        {
            //This op, or its parent, is not differentiable, so we will just assume its derivative is zero everywhere
            continue;
        }

        //Compute the derivative: d(op)/d(op.deps)
        auto depGrads = gradFunc(op, opGrad);

        //Add these to grads. If there is already an entry for one of the deps, then it has two parents.
        //we can just add this grad to the existing grad, because maths.
        foreach(d, g; zip(op.deps, depGrads))
        {
            auto currentGrad = grads.get(d, null);

            if(currentGrad is null)
            {
                grads[d] = g;
            }
            else
            {
                grads[d] = currentGrad + g;
            }
        }
    }

    auto errIdx = wrt.countUntil!(x => grads.get(x, null) is null);

    enforce(errIdx == -1, "Could not find wrt[" ~ errIdx.to!string ~ "] in the operation graph");

    return wrt.map!(x => grads[x]).array();
}

///
unittest
{
    import std.random : uniform;
    import dopt.core : evaluate;

    auto x = float32();
    auto y = x * x;
    auto gradY = grad(y, [x]);

    auto r = uniform(-100.0f, 100.0f);

    auto gradYwrtX = gradY.evaluate([
        x: Buffer([r])
    ])[0];

    assert(gradYwrtX.get!float[0] == r + r);
}

void registerGradient(string opName, Gradient g)
{
    enforce((opName in mGradients) is null, "A gradient is already registered for operation '" ~ opName ~ "'");

    mGradients[opName] = g;
}

void deregisterGradient(string opName)
{
    mGradients.remove(opName);
}

string[] listAllGradients()
{
    return mGradients.keys.dup;
}

private
{
    Gradient[string] mGradients;
}