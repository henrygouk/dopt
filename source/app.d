void main()
{
    import dopt.core;
    import std.stdio;

    auto a = float32([]);
    auto b = float32([]);
    auto c = float32([]);

    auto d = a * b + c;

    writeln("Expression: ", d);
    writeln("Gradient: ", d.grad([a]));
}