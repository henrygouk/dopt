module dopt.core.grads.basic;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional : toDelegate;

        registerGradient("transpose", toDelegate(&transposeGrad));
    }
}

private
{
    Operation[] transposeGrad(const(Operation) op, Operation parentGrad)
    {
        import std.algorithm : countUntil, map;
        import std.array : array;
        import std.range : iota;

        auto order = op
                    .attributes["order"]
                    .get!(const(size_t)[]);

        auto newOrder = iota(0, order.length)
                       .map!(x => cast(size_t)order.countUntil(x))
                       .array();

        return [parentGrad.transpose(newOrder)];
    }
}