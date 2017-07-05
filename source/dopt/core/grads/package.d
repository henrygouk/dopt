module dopt.core.grads;

import std.exception;

import dopt.core.grads.basic;
import dopt.core.grads.math;
import dopt.core.ops;

alias Gradient = Operation[] delegate(const(Operation) op, Operation parentGrad);

static this()
{
    dopt.core.grads.basic.initialize();
    dopt.core.grads.math.initialize();
}

Operation[] grad(const(Operation) objective, const(Operation)[] wrt)
{
    import std.algorithm : canFind, countUntil, map;
    import std.array : array;
    import std.conv : to;
    import std.range : retro, zip;

    enforce(objective.outputType.volume == 1, "The objective must have a volume of one");

    const(Operation)[] ops;

    void traverse(const(Operation) op)
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

    Operation[const(Operation)] grads;

    //TODO: when I implement a 'ones' operation, replace this line
    grads[objective] = float32([], [1.0f]);

    //Iterate through the operations in reverse order (reverse mode autodiff)
    foreach(op; ops.retro)
    {
        //Get the function that will let us compute the gradient of op w.r.t. its deps
        auto gradFunc = mGradients.get(op.opType, null);
        auto opGrad = grads[op];
        
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