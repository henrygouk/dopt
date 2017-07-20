/**
    This module enables operation graphs to be evaluated using CPU kernels.

    Author: Henry Gouk
*/
module dopt.core.cpu;

import std.exception;

import dopt.core;

/**
    Common interface for all CPU kernels.
*/
interface CPUKernel
{
    void execute(const(Operation) op, const(Buffer)[] inputs, Buffer output);
}

/**
    Convenience class that allows one to wrap a delegate and implement CPUKernel.
*/
class CPUKernelDelegate : CPUKernel
{
    public
    {
        this(void delegate(const(Operation), const(Buffer)[], Buffer) kern)
        {
            mKernel = kern;
        }

        void execute(const(Operation) op, const(Buffer)[] inputs, Buffer output)
        {
            mKernel(op, inputs, output);
        }
    }

    private
    {
        void delegate(const(Operation) op, const(Buffer)[], Buffer) mKernel;
    }
}

/**
    Registers a kernel for the specified operation.

    Params:
        opName = The name of the operation.
        kernel = A kernel that can execute operations of the type specified by opName.

    Throws:
        If there is already a kernel registered for the operation.
*/
void registerCPUKernel(string opName, CPUKernel kernel)
{
    enforce((opName in mKernels) is null, "A CPUKernel is already registered for the operation '" ~ opName ~ "'");

    mKernels[opName] = kernel;
}

/**
    Deregisters the kernel associated with the specified operation.

    Params:
        opName = The name of the operation that should have its kernel deregistered.
*/
void deregisterCPUKernel(string opName)
{
    mKernels.remove(opName);
}

/**
    Provides a list of operations for which a CPUKernel has been registered.

    Returns:
        An array of operation names.
*/
string[] listAllCPUOperations()
{
    return mKernels.keys.dup ~ ["variable", "reshape"];
}

/**
    Evaluates an several nodes from the operation graph.

    If the elements of $(D ops) have common dependencies, then each dependency is evaluated more than one. For this
    reason it is recommended that this overload is used when multiple nodes should be evaluated.

    Params:
        ops = The nodes of the operation graph that values should be computed for.
        args = A set of variable assignments.

    Returns:
        An array of $(D Buffer) objects, each containing the value of the corresponding element in $(D ops).
*/
Buffer[] evaluate(const(Operation)[] ops, Buffer[const(Operation)] args = null)
{
    //Toposort the operations by dependency
    const(Operation)[] sortedOps;

    void toposort(const(Operation) o)
    {
        import std.algorithm : canFind;

        foreach(d; o.deps)
        {
            toposort(d);
        }

        if(!sortedOps.canFind(o) && !args.keys.canFind(o))
        {
            sortedOps ~= o;
        }
    }

    foreach(o; ops)
    {
        toposort(o);
    }

    //Count the number of references to each operation
    int[const(Operation)] refCounts;

    foreach(o; sortedOps)
    {
        foreach(d; o.deps)
        {
            refCounts[d]++;
        }
    }

    //Start executing the operations
    Buffer[const(Operation)] results = args.dup;

    foreach(o; sortedOps)
    {
        import std.conv : to;
        import std.stdio : stdout, write, writeln;

        //Check for some easy optimizations
        if(o.opType == "variable" && !("variable" in mKernels))
        {
            results[o] = cast(Buffer)o.attributes["default"].get!Buffer;
            continue;
        }
        else if(o.opType == "reshape" && !("reshape" in mKernels))
        {
            results[o] = results[o.deps[0]];
            continue;
        }

        //Allocate a buffer for the output of this operation
        auto output = Buffer(new ubyte[o.outputType.volume * o.outputType.elementType.sizeOf()]);
        results[o] = output;

        //Get the input buffers
        Buffer[] inputs;

        foreach(d; o.deps)
        {
            inputs ~= results[d];
            refCounts[d]--;
        }

        //Execute the operation
        mKernels[o.opType].execute(o, inputs, output);

        foreach(d; o.deps)
        {
            //Remove the pointer to this buffer if we don't need it anymore
            //This will allow the GC to collect it at some point, if required
            if(refCounts[d] == 0)
            {
                results[d] = Buffer([]);
            }
        }
    }

    Buffer[] returnVals = new Buffer[ops.length];

    foreach(i, o; ops)
    {
        returnVals[i] = results[o];
    }

    return returnVals;
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

private
{
    CPUKernel[string] mKernels;
}