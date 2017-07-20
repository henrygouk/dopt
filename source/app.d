module app;

void main()
{
    import dopt.core;
    import std.stdio;

    writeln("Operations: ", listAllOperations());
    writeln("CPU Operations: ", listAllCPUOperations());
    writeln("Differentiable Operations: ", listAllGradients());
}