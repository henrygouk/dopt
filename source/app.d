void main()
{
    import dopt.core;
    import std.stdio;

    auto a = float32([3, 3], [1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f]);
    auto b = float32([3, 3], [1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f]);
    auto c = float32([3, 3], [2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f]);

    auto d = (a + b) * c;

    writeln(cast(float[])evaluate(d).get!(void[]));
}