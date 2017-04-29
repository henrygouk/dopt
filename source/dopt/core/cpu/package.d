module dopt.core.cpu;

import std.exception;

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

private
{
    CPUKernel[string] mKernels;
}