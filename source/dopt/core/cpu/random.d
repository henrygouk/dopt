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
        import std.random : uniform01;

        float[] arr = output.as!float[0 .. op.volume];
        foreach (ref v; arr)
            v = uniform01!float + float.epsilon; // by default CUDA's random is 0-1 including 1 but not zero, D is 0-1 including 0 but not one.
    }
}
