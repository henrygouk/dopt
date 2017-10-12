module dopt.core.ops.random;

import dopt.core.ops;
import dopt.core.types;

import std.array;
import std.functional;
import std.variant;

package
{
    void initialize()
    {
        registerOperation("uniform", OpDef(toDelegate(&verifyUniform), toDelegate(&judgeRandom)));
    }
}

private
{
    bool verifyUniform(Operation op)
    {
        auto shape = "shape" in op.attributes;

        return (shape !is null) && (shape.peek!(size_t[]) !is null);
    }

    TensorType judgeRandom(Operation op)
    {
        return TensorType(DataType.float32, op.attributes["shape"].get!(size_t[]));
    }
}

public
{
    Operation uniformSample(size_t[] shape, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("uniform", [], ["shape": Variant(shape)], mod, line);
    }
}