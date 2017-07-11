module dopt.core.ops.basic;

import dopt.core.ops;
import dopt.core.types;

import std.algorithm;
import std.array;
import std.functional;
import std.range;
import std.variant;

package
{
    void initialize()
    {
        registerOperation("slice", OpDef(toDelegate(&verifySlice), toDelegate(&judgeSlice)));
        registerOperation("pad", OpDef(toDelegate(&verifyPad), toDelegate(&judgePad)));
        registerOperation("reshape", OpDef(toDelegate(&verifyReshape), toDelegate(&judgeReshape)));
        registerOperation("transpose", OpDef(toDelegate(&verifyTranspose), toDelegate(&judgeTranspose)));
        registerOperation("repeat", OpDef(toDelegate(&verifyRepeat), toDelegate(&judgeRepeat)));
        registerOperation("variable", OpDef(toDelegate(&verifyVariable), toDelegate(&judgeVariable)));
    }
}

private
{
    bool verifySlice(const(Operation) op)
    {
        if(("start" in op.attributes) is null || ("stop" in op.attributes) is null)
        {
            return false;
        }

        auto startVar = op.attributes["start"];
        auto stopVar = op.attributes["stop"];

        if(startVar.peek!(const(size_t)[]) is null || stopVar.peek!(const(size_t)[]) is null)
        {
            return false;
        }

        auto start = startVar.get!(const(size_t)[]);
        auto stop = stopVar.get!(const(size_t)[]);

        return op.deps.length == 1
            && start.length == stop.length
            && start.length == op.deps[0].outputType.rank
            && zip(start, op.deps[0].outputType.shape).all!(x => x[0] < x[1])
            && zip(stop, op.deps[0].outputType.shape).all!(x => x[0] <= x[1])
            && zip(start, stop).all!(x => x[0] < x[1]);
    }

    TensorType judgeSlice(const(Operation) op)
    {
        auto start = op
                    .attributes["start"]
                    .get!(const(size_t)[]);

        auto stop = op
                   .attributes["stop"]
                   .get!(const(size_t)[]);

        auto shape = zip(start, stop)
                    .map!(x => x[1] - x[0])
                    .array();

        return TensorType(op.deps[0].outputType.elementType, shape);
    }

    bool verifyPad(const(Operation) op)
    {
        if(("before" in op.attributes) is null || ("after" in op.attributes) is null)
        {
            return false;
        }

        auto beforeVar = op.attributes["before"];
        auto afterVar = op.attributes["after"];

        if(beforeVar.peek!(const(size_t)[]) is null || afterVar.peek!(const(size_t)[]) is null)
        {
            return false;
        }

        auto before = beforeVar.get!(const(size_t)[]);
        auto after = afterVar.get!(const(size_t)[]);

        return op.deps.length == 1
            && before.length == after.length
            && before.length == op.deps[0].outputType.rank;
    }

    TensorType judgePad(const(Operation) op)
    {
        auto before = op
                     .attributes["before"]
                     .get!(const(size_t)[]);

        auto after = op
                    .attributes["after"]
                    .get!(const(size_t)[]);

        auto shape = zip(before, after, op.deps[0].outputType.shape)
                    .map!(x => x[0] + x[1] + x[2])
                    .array();

        return TensorType(op.deps[0].outputType.elementType, shape);
    }

    bool verifyReshape(const(Operation) op)
    {
        auto newShape = "shape" in op.attributes;

        return op.deps.length == 1
            && newShape !is null
            && newShape.peek!(const(size_t)[]) !is null
            && newShape.get!(const(size_t)[]).fold!((a, b) => a * b)(cast(size_t)1) == op.deps[0].outputType.volume;
    }

    TensorType judgeReshape(const(Operation) op)
    {
        return TensorType(op.deps[0].outputType.elementType, op.attributes["shape"].get!(const(size_t)[]));
    }

    bool verifyTranspose(const(Operation) op)
    {
        auto newOrder = "order" in op.attributes;

        return op.deps.length == 1
            && newOrder !is null
            && newOrder.peek!(const(size_t)[]) !is null
            && newOrder.get!(const(size_t)[]).dup.sort().equal(iota(0, op.deps[0].outputType.rank));
    }

    TensorType judgeTranspose(const(Operation) op)
    {
        auto order = op
                    .attributes["order"]
                    .get!(const(size_t)[]);

        auto newShape = order
                       .map!(x => op.deps[0].outputType.shape[x])
                       .array();

        return TensorType(op.deps[0].outputType.elementType, newShape);
    }

    bool verifyRepeat(const(Operation) op)
    {
        return op.deps.length == 1
            && ("repititions" in op.attributes) !is null;
    }

    TensorType judgeRepeat(const(Operation) op)
    {
        size_t reps = op.attributes["repititions"].get!size_t;

        return TensorType(op.deps[0].outputType.elementType, reps ~ op.deps[0].outputType.shape);
    }

    bool verifyVariable(const(Operation) op)
    {
        return op.deps.length == 0
            && ("type" in op.attributes) !is null
            && op.attributes["type"].peek!TensorType !is null;
    }

    TensorType judgeVariable(const(Operation) op)
    {
        return TensorType(op.attributes["type"].get!(const(TensorType)));
    }
}

public
{
    /**
    Produces a tensor that results from performing a slice operation similar to input[start .. stop].
    */
    Operation slice(const(Operation) input, const(size_t)[] start, const(size_t)[] stop,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("slice", [input], ["start": Variant(start), "stop": Variant(stop)], mod, line);
    }

    /**
    Extends the size of the input by padding it with zeros.
    */
    Operation pad(const(Operation) input, const(size_t)[] before, const(size_t)[] after,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("pad", [input], ["before": Variant(before), "after": Variant(after)], mod, line);
    }

    /**
    Essentially acts as a type casting operation, but for only the shape component of the TensorType.
    */
    Operation reshape(const(Operation) input, const(size_t)[] shape, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("reshape", [input], ["shape": Variant(shape)], mod, line);
    }

    /**
    Changes the order of the dimensions of the input tensor.
    */
    Operation transpose(const(Operation) input, const(size_t)[] order, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("transpose", [input], ["order": Variant(order)], mod, line);
    }

    /**
    Repeats a tensor, and increases the rank of the tensor by one in order to index these repititions.
    */
    Operation repeat(const(Operation) input, size_t repititions, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("repeat", [input], ["repititions": Variant(repititions)], mod, line);
    }

    /**
    Creates a variable of the given type.

    The resulting operation contains an attribute called "default", which is used as the default value of the variable.
    */
    Operation variable(TensorType type, void[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        auto bufSize = type.volume * sizeOf(type.elementType);

        if(defaultVal is null)
        {
            defaultVal = new ubyte[bufSize];
        }

        return createOperation("variable", [], ["type": Variant(type), "default": Variant(defaultVal)], mod, line);
    }

    Operation float32(const(size_t)[] size, float[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.float32, size), defaultVal, mod, line);
    }

    Operation int32(const(size_t)[] size, int[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.int32, size), defaultVal, mod, line);
    }
}