module dopt.core.cpu;

import std.exception;

import dopt.core;

interface CPUKernel
{
    void execute(const(Operation) op, const(Buffer)[] inputs, Buffer output);
}

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

void registerCPUKernel(string opName, CPUKernel kernel)
{
    enforce((opName in mKernels) is null, "A CPUKernel is already registered for the operation '" ~ opName ~ "'");

    mKernels[opName] = kernel;
}

void deregisterCPUKernel(string opName)
{
    mKernels.remove(opName);
}

string[] listAllCPUOperations()
{
    return mKernels.keys.dup ~ ["variable", "reshape"];
}

Buffer evaluate(const(Operation) op, Buffer[const(Operation)] args = null)
{
    return evaluate([op], args)[0];
}

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
        //writeln("Allocated buffer for operation ", (cast(void *)o).to!string);

        //Get the input buffers
        Buffer[] inputs;

        foreach(d; o.deps)
        {
            inputs ~= results[d];
            refCounts[d]--;
        }

        //Execute the operation
        //write("Executing operation ", (cast(void *)o).to!string, "...");
        //stdout.flush();

        mKernels[o.opType].execute(o, inputs, output);

        //writeln(" done");

        foreach(d; o.deps)
        {
            //Remove the pointer to this buffer if we don't need it anymore
            //This will allow the GC to collect it at some point, if required
            if(refCounts[d] == 0)
            {
                //writeln("Freeing buffer for operation ", (cast(void *)d).to!string);
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

private
{
    CPUKernel[string] mKernels;
}