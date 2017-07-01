module dopt.core.cpu;

import std.exception;
import std.variant;

import dopt.core;

interface CPUKernel
{
    void execute(const(Operation) op, const(void[])[] inputs, void[] output);
}

class CPUKernelDelegate : CPUKernel
{
    public
    {
        this(void delegate(const(Operation), const(void[])[], void[]) kern)
        {
            mKernel = kern;
        }

        void execute(const(Operation) op, const(void[])[] inputs, void[] output)
        {
            mKernel(op, inputs, output);
        }
    }

    private
    {
        void delegate(const(Operation) op, const(void[])[], void[]) mKernel;
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

string[] listAllCPUKernels()
{
    return mKernels.keys.dup;
}

Variant evaluate(const(Operation) op, Variant[const(Operation)] args = null)
{
    return evaluate([op], args)[0];
}

Variant[] evaluate(const(Operation)[] ops, Variant[const(Operation)] args = null)
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
    void[][const(Operation)] results;

    //Put the args into results
    foreach(k, v; args)
    {
        results[k] = v.get!(void[]);
    }

    foreach(o; sortedOps)
    {
        import std.conv : to;
        import std.stdio : stdout, write, writeln;

        //Check for some easy optimizations
        if(o.opType == "variable" && !("variable" in mKernels))
        {
            results[o] = cast(void[])o.attributes["default"].get!(void[]);
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
        //writeln("Allocated buffer for operation ", (cast(void *)o).to!string);

        //Get the input buffers
        void[][] inputs;

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
                results[d] = null;
            }
        }
    }

    Variant[] returnVals = new Variant[ops.length];

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