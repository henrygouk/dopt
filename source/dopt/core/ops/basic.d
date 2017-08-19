/**
    Contains operations for creating variables and manipulating shapes.

    Authors: Henry Gouk
*/
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
            && ("repetitions" in op.attributes) !is null;
    }

    TensorType judgeRepeat(const(Operation) op)
    {
        size_t reps = op.attributes["repetitions"].get!size_t;

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
        Slices the result of an operation.

        Params:
            input = The operation that should be sliced.
            start = The starting indices for each dimension.
            stop = The stopping indices for each dimension.

        Returns:
            The new $(D Operation).
    */
    Operation slice(const(Operation) input, const(size_t)[] start, const(size_t)[] stop,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("slice", [input], ["start": Variant(start), "stop": Variant(stop)], mod, line);
    }

    /**
        Pads the result of an operation with zeros in each dimension.

        Params:
            input = The operation that should be padded.
            before = The amount of padding that should be prepended for each dimension.
            after = The amount of padding that should be appended for each dimension.

        Returns:
            The new $(D Operation).
    */
    Operation pad(const(Operation) input, const(size_t)[] before, const(size_t)[] after,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("pad", [input], ["before": Variant(before), "after": Variant(after)], mod, line);
    }

    /**
        Allows one to cast an operation to a different shape with the same volume.

        Params:
            input = The operation to be reshaped.
            shape = The new shape.

        Returns:
            The new $(D Operation).
    */
    Operation reshape(const(Operation) input, const(size_t)[] shape, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("reshape", [input], ["shape": Variant(shape)], mod, line);
    }

    /**
        Reorders the dimensions of output of an operation.

        Params:
            input = The operation that should have its dimensions reordered.
            order = Determines how the dimensions are permuted.

        Returns:
            The new $(D Operation).
    */
    Operation transpose(const(Operation) input, const(size_t)[] order, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("transpose", [input], ["order": Variant(order)], mod, line);
    }

    /**
        Repeats the output of an operation the given number of times.

        A new dimension with a length of $(D repetitions) is added.

        Params:
            input = The operation to have its output repeated.
            repetitions = The number of repetitions to perform.

        Return:
            
    */
    Operation repeat(const(Operation) input, size_t repetitions, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("repeat", [input], ["repetitions": Variant(repetitions)], mod, line);
    }

    /**
        Creates a variable with the given type.

        If no default value is provided, then the variable will have a default value of all zeros. The default value is
        stored in the attributes["default"] field of the returned operation.

        Params:
            type = The type of the variable
            defaultVal = The default value of the variable. The array should store the elements in row major order.

        Returns:
            The newly created variable
    */
    Operation variable(TensorType type, void[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        auto bufSize = type.volume * sizeOf(type.elementType);

        if(defaultVal is null)
        {
            defaultVal = new ubyte[bufSize];
        }

        return createOperation("variable", [], ["type": Variant(type), "default": Variant(Buffer(defaultVal))], mod, line);
    }

    /**
        Creates a variable with the given shape and float32 elements.

        If no default value is provided, then the variable will have a default value of all zeros. The default value is
        stored in the attributes["default"] field of the returned operation.

        Params:
            size = The shape of the variable
            defaultVal = The default value of the variable. The array should store the elements in row major order.

        Returns:
            The newly created variable
    */
    Operation float32(const(size_t)[] size, float[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.float32, size), defaultVal, mod, line);
    }

    /**
        Creates a variable with the given shape and int32 elements.

        If no default value is provided, then the variable will have a default value of all zeros. The default value is
        stored in the attributes["default"] field of the returned operation.

        Params:
            size = The shape of the variable
            defaultVal = The default value of the variable. The array should store the elements in row major order.

        Returns:
            The newly created variable
    */
    Operation int32(const(size_t)[] size, int[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.int32, size), defaultVal, mod, line);
    }
}