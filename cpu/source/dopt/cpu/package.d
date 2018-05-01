/**
    This module enables operation graphs to be evaluated using CPU kernels.

    Authors: Henry Gouk
*/
module dopt.cpu;

import std.exception;

import dopt.core;

shared static this()
{
    import dopt.cpu.basic;
    import dopt.cpu.math;
    import dopt.cpu.nnet;
    import dopt.cpu.random;

    dopt.cpu.basic.initialize();
    dopt.cpu.math.initialize();
    dopt.cpu.nnet.initialize();
    dopt.cpu.random.initialize();

    import std.functional : toDelegate;
    defaultEvaluator = toDelegate(&evaluateCPU);
    defaultCompiler = (Operation[] ops) { return new CPUPlan(ops); };
    defaultVarAllocator = (size_t numBytes) { return new CPUBuffer(numBytes); };
    defaultArgAllocator = (size_t numBytes) { return new CPUBuffer(numBytes); };
}

/**
    Common interface for all CPU kernels.
*/
interface CPUKernel
{
    void execute(Operation op, const(void[])[] inputs, void[] output);
}

/**
    Convenience class that allows one to wrap a delegate and implement CPUKernel.
*/
class CPUKernelDelegate : CPUKernel
{
    public
    {
        this(void delegate(Operation, const(void[])[], void[]) kern)
        {
            mKernel = kern;
        }

        void execute(Operation op, const(void[])[] inputs, void[] output)
        {
            mKernel(op, inputs, output);
        }
    }

    private
    {
        void delegate(Operation op, const(void[])[], void[]) mKernel;
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
    return mKernels.keys.dup ~ ["constant", "variable", "reshape"];
}

class CPUBuffer : DeviceBuffer
{
    public
    {
        this(size_t len)
        {
            mBuffer = new ubyte[len];
        }

        this(void[] buf)
        {
            mBuffer = buf.dup;
        }

        override size_t numBytes() const
        {
            return mBuffer.length;
        }

        override void get(void[] buf) const
        {
            buf[] = mBuffer[];
        }

        override void set(const void[] buf)
        {
            mBuffer[] = buf[];
        }

        ubyte[] raw()
        {
            return cast(ubyte[])mBuffer;
        }
    }

    private
    {
        void[] mBuffer;
    }
}

class CPUPlan : Plan
{
    public
    {
        this(Operation[] outputs)
        {
            super(outputs);
        }
    }

    protected
    {
        override void executeImpl(DeviceBuffer[Operation] args, DeviceBuffer[] rets)
        {
            auto tmpRets = evaluateCPU(mOutputs, args);

            import std.range : zip;

            foreach(t, r; zip(tmpRets, rets))
            {
                r.set(t);
            }
        }
    }
}

/**
    Evaluates an several nodes from the operation graph using the CPU.

    If the elements of $(D ops) have common dependencies, then each dependency is evaluated only once. For this
    reason it is recommended that this overload is used when multiple nodes should be evaluated.

    Params:
        ops = The nodes of the operation graph that values should be computed for.
        args = A set of variable assignments.

    Returns:
        An array of $(D Buffer) objects, each containing the value of the corresponding element in $(D ops).
*/
DeviceBuffer[] evaluateCPU(Operation[] ops, DeviceBuffer[Operation] args = null)
{
    import std.algorithm : canFind, filter;
    import std.array : array;

    //Toposort the operations by dependency
    Operation[] sortedOps = topologicalSort(ops)
                                  .filter!(x => !canFind(args.keys, x))
                                  .array();

    //Count the number of references to each operation
    int[Operation] refCounts;

    foreach(o; ops)
    {
        refCounts[o]++;
    }

    foreach(o; sortedOps)
    {
        foreach(d; o.deps)
        {
            refCounts[d]++;
        }
    }

    //Start executing the operations
    ubyte[][Operation] results;

    foreach(k, v; args)
    {
        results[k] = v.get!ubyte();
    }

    foreach(o; sortedOps)
    {
        import std.conv : to;
        import std.stdio : stdout, write, writeln;

        //Check for some easy optimizations
        if(o.opType == "variable" && !("variable" in mKernels))
        {
            results[o] = o.value.get!ubyte;
            continue;
        }
        else if(o.opType == "constant" && !("constant" in mKernels))
        {
            results[o] = o.value.get!ubyte;
            continue;
        }
        else if(o.opType == "reshape" && !("reshape" in mKernels))
        {
            results[o] = results[o.deps[0]];
            continue;
        }

        //Allocate a buffer for the output of this operation
        auto output = new ubyte[o.outputType.volume * o.outputType.elementType.sizeOf()];
        results[o] = output;

        //Get the input buffers
        ubyte[][] inputs;

        foreach(d; o.deps)
        {
            inputs ~= results[d];
            refCounts[d]--;
        }

        //Execute the operation
        auto kern = mKernels.get(o.opType, null);

        if(kern is null)
        {
            throw new Exception("No CPU kernel registered for operation " ~ o.opType);
        }

        kern.execute(o, cast(const(void[])[]) inputs, cast(void[])output);

        foreach(d; o.deps)
        {
            //Remove the pointer to this buffer if we don't need it anymore
            //This will allow the GC to collect it at some point, if required
            if(refCounts[d] == 0)
            {
                results[d] = null;
            }
        }
    }

    DeviceBuffer[] returnVals = new DeviceBuffer[ops.length];

    foreach(i, o; ops)
    {
        returnVals[i] = new CPUBuffer(results[o]);
    }

    return returnVals;
}

private
{
    CPUKernel[string] mKernels;
}