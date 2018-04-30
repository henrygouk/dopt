/**
    This package contains the framework for constructing and executing operation graphs.

    $(UL
        $(LI $(D dopt.core.ops) provides functions for constructing nodes in the operation graph.)
        $(LI $(D dopt.core.grads) provides functions for computing the derivatives of operations.)
        $(LI $(D dopt.core.cpu) contains a backend that executes operation graphs using the CPU.)
        $(LI $(D dopt.core.cuda) contains a backend that executes operation graphs using a CUDA enabled GPU.)
    )

    Authors: Henry Gouk
*/
module dopt.core;

public
{
    import dopt.core.grads;
    import dopt.core.ops;
    import dopt.core.types;
}

alias Evaluator = DeviceBuffer[] delegate(Operation[] ops, DeviceBuffer[Operation] args);
alias Compiler = Plan delegate(Operation[] ops);
alias Allocator = DeviceBuffer delegate(size_t numBytes);

private __gshared Evaluator mDefaultEvaluator;
private __gshared Compiler mDefaultCompiler;
private __gshared Allocator mDefaultAllocator;

Evaluator defaultEvaluator()
{
    return mDefaultEvaluator;
}

void defaultEvaluator(Evaluator de)
{
    mDefaultEvaluator = de;
}

Compiler defaultCompiler()
{
    return mDefaultCompiler;
}

void defaultCompiler(Compiler de)
{
    mDefaultCompiler = de;
}

Allocator defaultAllocator()
{
    return mDefaultAllocator;
}

void defaultAllocator(Allocator da)
{
    mDefaultAllocator = da;
}

shared static this()
{
    import std.functional : toDelegate;

    dopt.core.ops.initialize();
    dopt.core.grads.initialize();
}

/**
    Evaluates a several nodes from the operation graph.

    Params:
        ops = The nodes of the operation graph that values should be computed for.
        args = A set of variable assignments.

    Returns:
        An array of $(D DeviceBuffer) objects, each containing the value of the corresponding element in $(D ops).
*/
DeviceBuffer[] evaluate(Operation[] ops, DeviceBuffer[Operation] args = null)
{
    return mDefaultEvaluator(ops, args);
}

/**
    Evaluates an operation graph with a single root node.

    This overload is here for convenience. Internally, the multi-output version of evaluate is called.

    Params:
        op = The root node of the operation graph.
        args = A set of variable assignments.

    Returns:
        A $(D Buffer) containing the result of the computation.
*/
DeviceBuffer evaluate(Operation op, DeviceBuffer[Operation] args = null)
{
    return evaluate([op], args)[0];
}

/**
    Compile an Operation graph into a reusable execution plan.

    This can be useful in the case where the function might need to be evaluated multiple times, as it will avoid
    repeating initialisation and optimisation procedures.

    Params:
        outputs = The output nodes of the Operation graph.
    
    Returns:
        A $(D Plan) that can be executed.
*/
Plan compile(Operation[] outputs)
{
    return mDefaultCompiler(outputs);
}

DeviceBuffer allocate(size_t numBytes)
{
    return mDefaultAllocator(numBytes);
}

DeviceBuffer buffer(void[] vals)
{
    auto buf = allocate(vals.length);
    buf.set(vals);

    return buf;
}

class Plan
{
    public
    {
        this(Operation[] outputs)
        {
            import std.array : array;

            mOutputs = outputs.array();
        }

        /**
            Executes the plan.

            Params:
                args = A set of variable assignments.
        */
        DeviceBuffer[] execute(DeviceBuffer[Operation] args = null)
        {
            auto rets = new DeviceBuffer[mOutputs.length];

            foreach(i, o; mOutputs)
            {
                rets[i] = allocate(o.outputType.volume * o.outputType.elementType.sizeOf());
            }

            execute(args, rets);

            return rets;
        }

        ///
        void execute(DeviceBuffer[Operation] args, DeviceBuffer[] rets)
        {
            executeImpl(args, rets);
        }
    }

    protected
    {
        Operation[] mOutputs;

        abstract void executeImpl(DeviceBuffer[Operation] args, DeviceBuffer[] rets);
    }
}