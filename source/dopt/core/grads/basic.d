module dopt.core.grads.basic;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional : toDelegate;

        registerGradient("transpose", toDelegate(&transposeGrad));
        registerGradient("slice", toDelegate(&sliceGrad));
        registerGradient("pad", toDelegate(&padGrad));
        registerGradient("reshape", toDelegate(&reshapeGrad));
        registerGradient("repeat", toDelegate(&repeatGrad));
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

    Operation[] sliceGrad(const(Operation) op, Operation parentGrad)
    {
        auto before = op.attributes["start"].get!(const(size_t)[]);
        auto after = op.deps[0].outputType.shape.dup;
        after[] -= op.attributes["stop"].get!(const(size_t)[])[];

        return [parentGrad.pad(before, after)];
    }

    //Test the sliceGrad function
    unittest
    {
        import dopt.core;

        auto a = float32([4, 4], [
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f]);

        auto b = float32([4, 4], [
            5.0f, 6.0f, 7.0f, 8.0f, 5.0f, 6.0f, 7.0f, 8.0f,
            5.0f, 6.0f, 7.0f, 8.0f, 5.0f, 6.0f, 7.0f, 8.0f]);

        auto c = slice(a * b, [1, 1], [2, 2]);

        import std.algorithm : equal;

        assert(evaluate(grad(c, [a]))[0].as!float.equal(
            [0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
    }

    Operation[] padGrad(const(Operation) op, Operation parentGrad)
    {
        auto start = op.attributes["before"].get!(const(size_t)[]);
        auto stop = op.deps[0].outputType.shape.dup;
        stop[] += start[];

        return [parentGrad.slice(start, stop)];
    }

    Operation[] reshapeGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad.reshape(op.deps[0].outputType.shape)];
    }

    Operation[] repeatGrad(const(Operation) op, Operation parentGrad)
    {
        import std.array : array;
        import std.range : iota, roundRobin;

        auto reps = op.attributes["repetitions"].get!(const(size_t)[]);
        
        //Add some new dimensions that explicitly represent the repetitions
        auto tmpShape = roundRobin(reps, op.deps[0].shape).array();
        auto tmp = parentGrad.reshape(tmpShape);

        //Sum over these dimensions
        return [tmp.sum(iota(0, tmpShape.length, 2).array())];
    }
}