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
    assert(0, "Not implemented");
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