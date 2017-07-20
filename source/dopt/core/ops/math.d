/**
    Contains constructors for common math operations.

    Authors: Henry Gouk
*/
module dopt.core.ops.math;

import std.algorithm;
import std.conv;
import std.functional;
import std.range;

import dopt.core.ops;
import dopt.core.types;

package
{
    void initialize()
    {
        void registerPointwiseBinary(string opName)
        {
            bool verifier(const(Operation) op)
            {
                return op.deps.length == 2 && op.deps[0].outputType == op.deps[1].outputType;
            }

            TensorType judge(const(Operation) op)
            {
                return TensorType(op.deps[0].outputType);
            }

            registerOperation(opName, OpDef(&verifier, &judge));
        }

        void registerPointwiseUnary(string opName)
        {
            bool verifier(const(Operation) op)
            {
                return true;
            }

            TensorType judge(const(Operation) op)
            {
                return TensorType(op.deps[0].outputType);
            }

            registerOperation(opName, OpDef(&verifier, &judge));
        }

        foreach(opName; chain(arith, comp, binfunc))
        {
            registerPointwiseBinary(opName);
        }

        foreach(opName; unfunc)
        {
            registerPointwiseUnary(opName);
        }

        registerOperation("matmul", OpDef(toDelegate(&verifyMatmul), toDelegate(&judgeMatmul)));
        registerOperation("sum", OpDef(toDelegate(&verifySum), toDelegate(&judgeSum)));
    }
}

private
{
    immutable string[] arith = ["add", "sub", "mul", "div"];
    immutable string[] comp = ["lt", "lte", "gt", "gte", "eq", "neq"];
    immutable string[] binfunc = ["max", "min", "pow"];
    immutable string[] unfunc = ["neg", "abs", "sgn", "exp", "log", "sqrt"];

    string createAllCtors()
    {
        string createOpCtor(string opName, size_t numDeps)
        {
            auto params = iota(0, numDeps)
                        .map!(x => "const(Operation) p" ~ x.to!string)
                        .joiner(", ")
                        .to!string();

            auto args = iota(0, numDeps)
                    .map!(x => "p" ~ x.to!string)
                    .joiner(", ")
                    .to!string;

            return "
                    Operation " ~ opName ~ "(" ~ params ~ ", string mod = __MODULE__, size_t line = __LINE__)
                    {
                        return createOperation(\"" ~ opName ~ "\", [" ~ args ~ "], null, mod, line);
                    }
                ";
        }

        string binctors = chain(arith, comp, binfunc)
                         .map!(x => createOpCtor(x, 2))
                         .joiner("\n")
                         .to!string;

        auto unctors = unfunc
                      .map!(x => createOpCtor(x, 1))
                      .joiner("\n")
                      .to!string;

        return binctors ~ unctors;
    }

    bool verifyMatmul(const(Operation) op)
    {
        return op.deps.length == 2
            && op.deps[0].outputType.rank == 2
            && op.deps[1].outputType.rank == 2
            && op.deps[0].outputType.elementType == op.deps[1].outputType.elementType
            && op.deps[0].outputType.shape[1] == op.deps[1].outputType.shape[0];
    }

    TensorType judgeMatmul(const(Operation) op)
    {
        return TensorType(op.deps[0].outputType.elementType,
            [op.deps[0].outputType.shape[0], op.deps[1].outputType.shape[1]]);
    }

    bool verifySum(const(Operation) op)
    {
        if(("rank" in op.attributes) is null || op.attributes["rank"].peek!size_t is null)
        {
            return false;
        }

        return op.attributes["rank"].get!size_t < op.deps[0].outputType.rank;
    }

    TensorType judgeSum(const(Operation) op)
    {
        auto t = op.deps[0].outputType;

        return TensorType(t.elementType, t.shape[0 .. op.attributes["rank"].get!size_t]);
    }
}

mixin(createAllCtors());

Operation matmul(const(Operation) lhs, const(Operation) rhs, string mod = __MODULE__, size_t line = __LINE__)
{
    return createOperation("matmul", [lhs, rhs], null, mod, line);
}

Operation sum(const(Operation) op, size_t newRank = 0, string mod = __MODULE__, size_t line = __LINE__)
{
    import std.variant : Variant;
    
    return createOperation("sum", [op], ["rank": Variant(newRank)], mod, line);
}