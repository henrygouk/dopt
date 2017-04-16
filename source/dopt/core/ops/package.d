module dopt.core.ops;

import std.array;
import std.exception;
import std.variant;

import dopt.core;

public
{
    import dopt.core.ops.math;
}

alias Verifier = bool delegate(const Operation);
alias Judge = TensorType delegate(const(Operation));

struct OpDef
{
    Verifier verifier;
    Judge judge;
}

class Operation
{
    public
    {
        @property string opType() const
        {
            return mOpType;
        }

        @property const(TensorType) outputType() const
        {
            return mOutputType;
        }

        @property const(Operation)[] deps() const
        {
            return mDeps;
        }

        @property const(Variant[string]) attributes() const
        {
            return mAttributes;
        }
    }

    private
    {
        string mOpType;
        string mModule;
        size_t mLine;
        const(Operation)[] mDeps;
        const(Variant[string]) mAttributes;
        const(TensorType) mOutputType;

        this(string opType, const(Operation)[] deps, const(Variant[string]) attribs, string mod, size_t line)
        {
            mOpType = opType;
            mDeps = deps.array;
            mAttributes = attribs.dup;
            mModule = mod;
            mLine = line;

            mOutputType = makeJudgement(this);
        }
    }
}

void registerOperation(string name, OpDef def)
{
    enforce((name in mOpDefs) is null, "There is already an operation registered with the name '" ~ name ~ "'");

    mOpDefs[name] = def;
}

string[] listAllOperations()
{
    return mOpDefs.keys.dup;
}

Operation createOperation(string opType, const(Operation)[] deps = [], const(Variant[string]) attribs = null,
    string mod = __MODULE__, size_t line = __LINE__)
{
    enforce(opType in mOpDefs,
        "Cannot create operation because there is no operation definition registered with the name '" ~ opType ~ "'");

    auto op = new Operation(opType, deps, attribs, mod, line);

    enforce(mOpDefs[opType].verifier(op), "Operation failed verification");

    return op;
}

private
{
    OpDef[string] mOpDefs;

    TensorType makeJudgement(const(Operation) op)
    {
        auto def = op.opType in mOpDefs;

        enforce(def !is null, "Cannot make judgement for unknown operation '" ~ op.opType() ~ "'");

        return def.judge(op);
    }
}