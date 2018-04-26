/**
    Contains functions for creating variable nodes and subsequently manipulating their shapes.

    Authors: Henry Gouk
*/
module dopt.core.ops.basic;

import dopt.core.ops;
import dopt.core.types;

import std.algorithm;
import std.array;
import std.exception;
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
        registerOperation("constant", OpDef(toDelegate(&verifyVariable), toDelegate(&judgeVariable)));
    }
}

private
{
    bool verifySlice(Operation op)
    {
        if(("start" in op.attributes) is null || ("stop" in op.attributes) is null)
        {
            return false;
        }

        auto startVar = op.attributes["start"];
        auto stopVar = op.attributes["stop"];

        if(startVar.peek!(size_t[]) is null || stopVar.peek!(size_t[]) is null)
        {
            return false;
        }

        auto start = startVar.get!(size_t[]);
        auto stop = stopVar.get!(size_t[]);

        return op.deps.length == 1
            && start.length == stop.length
            && start.length == op.deps[0].outputType.rank
            && zip(start, op.deps[0].outputType.shape).all!(x => x[0] < x[1])
            && zip(stop, op.deps[0].outputType.shape).all!(x => x[0] <= x[1])
            && zip(start, stop).all!(x => x[0] < x[1]);
    }

    TensorType judgeSlice(Operation op)
    {
        auto start = op
                    .attributes["start"]
                    .get!(size_t[]);

        auto stop = op
                   .attributes["stop"]
                   .get!(size_t[]);

        auto shape = zip(start, stop)
                    .map!(x => x[1] - x[0])
                    .array();

        return TensorType(op.deps[0].outputType.elementType, shape);
    }

    bool verifyPad(Operation op)
    {
        if(("before" in op.attributes) is null || ("after" in op.attributes) is null)
        {
            return false;
        }

        auto beforeVar = op.attributes["before"];
        auto afterVar = op.attributes["after"];

        if(beforeVar.peek!(size_t[]) is null || afterVar.peek!(size_t[]) is null)
        {
            return false;
        }

        auto before = beforeVar.get!(size_t[]);
        auto after = afterVar.get!(size_t[]);

        return op.deps.length == 1
            && before.length == after.length
            && before.length == op.deps[0].outputType.rank;
    }

    TensorType judgePad(Operation op)
    {
        auto before = op
                     .attributes["before"]
                     .get!(size_t[]);

        auto after = op
                    .attributes["after"]
                    .get!(size_t[]);

        auto shape = zip(before, after, op.deps[0].outputType.shape)
                    .map!(x => x[0] + x[1] + x[2])
                    .array();

        return TensorType(op.deps[0].outputType.elementType, shape);
    }

    bool verifyReshape(Operation op)
    {
        auto newShape = "shape" in op.attributes;

        return op.deps.length == 1
            && newShape !is null
            && newShape.peek!(size_t[]) !is null
            && newShape.get!(size_t[]).fold!((a, b) => a * b)(cast(size_t)1) == op.deps[0].outputType.volume;
    }

    TensorType judgeReshape(Operation op)
    {
        return TensorType(op.deps[0].outputType.elementType, op.attributes["shape"].get!(size_t[]));
    }

    bool verifyTranspose(Operation op)
    {
        auto newOrder = "order" in op.attributes;

        return op.deps.length == 1
            && newOrder !is null
            && newOrder.peek!(size_t[]) !is null
            && newOrder.get!(size_t[]).dup.sort().equal(iota(0, op.deps[0].outputType.rank));
    }

    TensorType judgeTranspose(Operation op)
    {
        auto order = op
                    .attributes["order"]
                    .get!(size_t[]);

        auto newShape = order
                       .map!(x => op.deps[0].outputType.shape[x])
                       .array();

        return TensorType(op.deps[0].outputType.elementType, newShape);
    }

    bool verifyRepeat(Operation op)
    {
        if(("repetitions" in op.attributes) is null)
        {
            return false;
        }

        auto reps = op.attributes["repetitions"].get!(size_t[]);

        return op.deps.length == 1
            && reps.length == op.deps[0].rank
            && reps.all!(x => x > 0);
    }

    TensorType judgeRepeat(Operation op)
    {
        auto reps = op.attributes["repetitions"].get!(size_t[]);
        auto shape = op.deps[0].shape.dup;
        shape[] *= reps[];

        return TensorType(op.deps[0].elementType, shape);
    }

    bool verifyVariable(Operation op)
    {
        return op.deps.length == 0
            && ("type" in op.attributes) !is null
            && op.attributes["type"].peek!TensorType !is null;
    }

    TensorType judgeVariable(Operation op)
    {
        return op.attributes["type"].get!TensorType;
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
    Operation slice(Operation input, size_t[] start, size_t[] stop,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("slice", [input], ["start": Variant(start), "stop": Variant(stop)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;

        auto s1 = int32([3, 3], [
            1, 2, 3,
            4, 5, 6,
            7, 8, 9
        ]).slice([1, 1], [3, 3]);

        assert(s1.evaluate().as!int == [
            5, 6,
            8, 9
        ]);
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
    Operation pad(Operation input, size_t[] before, size_t[] after,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("pad", [input], ["before": Variant(before), "after": Variant(after)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;

        auto p1 = int32([1, 1], [3]).pad([2, 1], [3, 3]);

        assert(p1.evaluate().as!int == [
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 3, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ]);
    }

    /**
        Allows one to cast an operation to a different shape with the same volume.

        Params:
            input = The operation to be reshaped.
            shape = The new shape.

        Returns:
            The new $(D Operation).
    */
    Operation reshape(Operation input, size_t[] shape, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("reshape", [input], ["shape": Variant(shape)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;

        auto r1 = float32([2, 2], [1.0f, 2.0f, 3.0f, 4.0f]).reshape([1, 4]);

        assert(r1.shape == [1, 4]);
        assert(r1.evaluate().as!float == [1.0f, 2.0f, 3.0f, 4.0f]);
    }

    /**
        Reorders the dimensions of output of an operation.

        Params:
            input = The operation that should have its dimensions reordered.
            order = Determines how the dimensions are permuted.

        Notes:
            Currently only implemented for rank 2 tensors.

        Returns:
            The new $(D Operation).
    */
    Operation transpose(Operation input, size_t[] order, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("transpose", [input], ["order": Variant(order)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;

        auto t1 = float32([2, 2], [1.0f, 2.0f, 3.0f, 4.0f]).transpose([1, 0]);

        assert(t1.evaluate().as!float == [1.0f, 3.0f, 2.0f, 4.0f]);
    }

    /**
        Repeats the output of an operation along each axis the given number of times.

        Params:
            input = The operation to have its output repeated.
            repetitions = The number of repetitions to perform along each axis.

        Return:
            The new $(D Operation).
    */
    Operation repeat(Operation input, size_t[] repetitions, string mod = __MODULE__,
        size_t line = __LINE__)
    {
        enforce(repetitions.length == input.rank,
            "The length of repetitions must be the same as the rank of the input.");
        
        return createOperation("repeat", [input], ["repetitions": Variant(repetitions)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;
        
        auto r1 = float32([1, 1], [3.0f]).repeat([2, 3]);
        auto r2 = float32([2, 2], [1.0f, 2.0f, 3.0f, 4.0f]).repeat([3, 2]);

        assert(r1.evaluate().as!float == [
            3.0f, 3.0f, 3.0f,
            3.0f, 3.0f, 3.0f
        ]);

        assert(r2.evaluate().as!float == [
            1.0f, 2.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 3.0f, 4.0f,
            1.0f, 2.0f, 1.0f, 2.0f,
            3.0f, 4.0f, 3.0f, 4.0f
        ]);
    }

    /**
        Repeats the output of an operation the given number of times.

        A new dimension is added, allowing one to index each of these repetitions.

        Params:
            input = The operation to have its output repeated.
            repetitions = The number of repetitions to perform.
        
        Return:
            The new $(D Operation).
    */
    Operation repeat(Operation input, size_t repetitions, string mod = __MODULE__, size_t line = __LINE__)
    {
        auto vec = input.reshape([1, input.volume]);

        import std.range : drepeat = repeat;
        import std.array : array;

        auto pattern = float32Constant([repetitions, 1], drepeat(1.0f, repetitions).array());
        auto r = pattern.matmul(vec);
        
        return r.reshape([repetitions] ~ input.shape, mod, line);
    }

    ///
    unittest
    {
        import dopt.core : evaluate;

        auto r1 = float32([2], [1.0f, 2.0f]).repeat(3);

        assert(r1.evaluate().as!float == [
            1.0f, 2.0f,
            1.0f, 2.0f,
            1.0f, 2.0f
        ]);
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
        else
        {
            enforce(defaultVal.length == bufSize, "The length of defaultVal does not match type.volume.");
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
    Operation float32(size_t[] size = [], float[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.float32, size), defaultVal, mod, line);
    }

    ///
    Operation float32(float defaultVal, string mod = __MODULE__, size_t line = __LINE__)
    {
        return float32([], [defaultVal], mod, line);
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
    Operation int32(size_t[] size = [], int[] defaultVal = null, string mod = __MODULE__, size_t line = __LINE__)
    {
        return variable(TensorType(DataType.int32, size), defaultVal, mod, line);
    }

    ///
    Operation int32(int defaultVal, string mod = __MODULE__, size_t line = __LINE__)
    {
        return int32([], [defaultVal], mod, line);
    }

    /**
        Creates a constant with the given type.

        Params:
            type = The type of the constant
            val = The value of the constant. The array should store the elements in row major order.

        Returns:
            The newly created constant
    */
    Operation constant(TensorType type, void[] val, string mod = __MODULE__, size_t line = __LINE__)
    {
        auto bufSize = type.volume * sizeOf(type.elementType);

        if(val is null)
        {
            val = new ubyte[bufSize];
        }
        else
        {
            enforce(val.length == bufSize, "The length of val does not match type.volume.");
        }

        return createOperation("constant", [], ["type": Variant(type), "default": Variant(Buffer(val))], mod, line);
    }

    /**
        Creates a constant with the given shape and float32 values.

        Params:
            size = The shape of the constant
            val = The value of the constant. The array should store the elements in row major order.

        Returns:
            The newly created constant
    */
    Operation float32Constant(size_t[] size, float[] val, string mod = __MODULE__, size_t line = __LINE__)
    {
        return constant(TensorType(DataType.float32, size), val, mod, line);
    }

    ///
    Operation float32Constant(float val, string mod = __MODULE__, size_t line = __LINE__)
    {
        return float32Constant([], [val], mod, line);
    }

    /**
        Creates a constant with the given shape and int32 values.

        Params:
            size = The shape of the constant
            val = The value of the constant. The array should store the elements in row major order.

        Returns:
            The newly created constant
    */
    Operation int32Constant(size_t[] size, int[] val, string mod = __MODULE__, size_t line = __LINE__)
    {
        return constant(TensorType(DataType.int32, size), val, mod, line);
    }

    ///
    Operation int32Constant(int val, string mod = __MODULE__, size_t line = __LINE__)
    {
        return int32Constant(val, mod, line);
    }
}