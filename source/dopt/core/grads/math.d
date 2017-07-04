module dopt.core.grads.math;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional;
        
        registerGradient("add", toDelegate(&addGrad));
        registerGradient("sub", toDelegate(&subGrad));
        registerGradient("mul", toDelegate(&mulGrad));
        registerGradient("div", toDelegate(&divGrad));
    }
}

private
{
    Operation[] addGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad, parentGrad];
    }

    Operation[] subGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad, neg(parentGrad)];
    }

    Operation[] mulGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad * op.deps[1], parentGrad * op.deps[0]];
    }

    Operation[] divGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            parentGrad / op.deps[1],
            neg(parentGrad * op.deps[0]) / (op.deps[1] * op.deps[1])
        ];
    }
}