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
    import dopt.core.cpu;
    import dopt.core.cuda;
    import dopt.core.grads;
    import dopt.core.ops;
    import dopt.core.types;
}

alias Evaluator = Buffer[] delegate(const(Operation)[] ops, Buffer[const(Operation)] args);

__gshared Evaluator defaultEvaluator;

shared static this()
{
    import std.functional : toDelegate;

    dopt.core.ops.initialize();
    dopt.core.grads.initialize();
    dopt.core.cpu.initialize();

    try
    {
        dopt.core.cuda.initialize();
        defaultEvaluator = toDelegate(&evaluateCUDA);
    }
    catch(Exception e)
    {
        defaultEvaluator = toDelegate(&evaluateCPU);
    }
}

/**
    Evaluates a several nodes from the operation graph.

    Params:
        ops = The nodes of the operation graph that values should be computed for.
        args = A set of variable assignments.

    Returns:
        An array of $(D Buffer) objects, each containing the value of the corresponding element in $(D ops).
*/
Buffer[] evaluate(const(Operation)[] ops, Buffer[const(Operation)] args = null)
{
    return defaultEvaluator(ops, args);
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
Buffer evaluate(const(Operation) op, Buffer[const(Operation)] args = null)
{
    return evaluate([op], args)[0];
}