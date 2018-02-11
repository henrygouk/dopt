module dopt.core.cpu.random;

import dopt.core;

package
{
    void initialize()
    {
        import std.functional : toDelegate;

        registerCPUKernel("uniform", new CPUKernelDelegate(toDelegate(&uniform)));
    }
}

private
{
    void uniform(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        import std.random : uniform;

        ubyte[] arr = output.as!ubyte[0 .. op.volume];
        foreach (ref v; arr)
            v = uniform!ubyte;
    }
}
