module dopt.core.cpu.math;

import dopt.core;

import std.algorithm;
import std.conv;
import std.functional;
import std.math;
import std.range;

package
{
    void initialize()
    {
        mixin(generateRegistrations());

        registerCPUKernel("matmul", new CPUKernelDelegate(toDelegate(&matmulKernel)));
        registerCPUKernel("sum", new CPUKernelDelegate(toDelegate(&sumKernel)));
        registerCPUKernel("maxElement", new CPUKernelDelegate(toDelegate(&maxElementKernel)));
        registerCPUKernel("argmin", new CPUKernelDelegate(toDelegate(&argminKernel)));
    }

    mixin(generateKernels());
}

private
{
    T expCast(T)(T t)
    {
        static if(is(T : int))
        {
            return cast(int)exp(cast(float)t);
        }
        else
        {
            return exp(t);
        }
    }

    T sqrtCast(T)(T t)
    {
        static if(is(T : int))
        {
            return cast(int)sqrt(cast(float)t);
        }
        else
        {
            return sqrt(t);
        }
    }

    T atan2(T)(T a, T b)
    {
        static if(is(T : int))
        {
            return cast(int)std.math.atan2(cast(float)a, cast(float)b);
        }
        else
        {
            return std.math.atan2(a, b);
        }
    }

    T sgn(T)(T t)
    {
        return cast(T)((t > 0) - (t < 0));
    }

    void matmulKernel(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        if(op.outputType.elementType == DataType.float32)
        {
            auto ashape = op.deps[0].outputType.shape;
            auto bshape = op.deps[1].outputType.shape;

            import cblas;

            gemm(Order.RowMajor, Transpose.NoTrans, Transpose.NoTrans,
                cast(int)ashape[0], cast(int)bshape[1], cast(int)ashape[1], 1.0, cast(float *)inputs[0].as!float.ptr,
                cast(int)ashape[1], cast(float *)inputs[1].as!float.ptr, cast(int)bshape[1], 0,
                cast(float *)output.as!float.ptr, cast(int)bshape[1]);
        }
        else
        {
            throw new Exception("Not implemented.");
        }
    }

    void sumKernel(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        void run(T)()
        {
            import std.algorithm : fold, sort;

            void process(const(T)[] inbuf, T[] outbuf, size_t highstride, size_t lowstride)
            {
                import std.array : array;
                import std.range : iota;
                import std.parallelism : parallel;

                foreach(o; iota(0, outbuf.length / lowstride).array().parallel)
                {
                    if(lowstride == 1)
                    {
                        outbuf[o] = inbuf[o * highstride .. (o + 1) * highstride]
                                   .fold!((a, b) => a + b)(cast(T)0);
                    }
                    else
                    {
                        outbuf[o * lowstride .. (o + 1) * lowstride] = 0;

                        for(size_t i = 0; i < highstride / lowstride; i++)
                        {
                            outbuf[o * lowstride .. (o + 1) * lowstride] +=
                                inbuf[o * highstride + i * lowstride .. o * highstride + (i + 1) * lowstride];
                        }
                    }
                }
            }

            auto axes = op.attributes["axes"].get!(size_t[]);
            auto shape = op.deps[0].shape.dup;

            auto inbuf = inputs[0].as!T;
            T[] outbuf;

            foreach(axis; axes)
            {
                //auto axis = axes[0];
                auto newvol = shape.fold!((a, b) => a * b)(size_t(1)) / shape[axis];
                size_t lowstride;
                
                if(axis == shape.length - 1)
                {
                    lowstride = 1;
                }
                else
                {
                    lowstride = shape[axis + 1 .. $].fold!((a, b) => a * b)(size_t(1));
                }

                size_t highstride = lowstride * shape[axis];

                outbuf = new T[newvol];
                process(inbuf, outbuf, highstride, lowstride);
                inbuf = outbuf;

                shape[axis] = 1;
            }

            output.as!T[] = outbuf[];
        }

        switch(op.outputType.elementType)
        {
            case DataType.float32:
                run!float();
                break;

            case DataType.int32:
                run!int();
                break;

            default:
                throw new Exception("Not implemented.");
        }
    }

    void maxElementKernel(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        void run(T)()
        {
            import std.algorithm : fold, max, sort;
            import std.parallelism : parallel;

            void process(const(T)[] inbuf, T[] outbuf, size_t highstride, size_t lowstride)
            {
                foreach(o; iota(0, outbuf.length / lowstride).array().parallel)
                {
                    outbuf[o * lowstride .. (o + 1) * lowstride] = -T.max;

                    for(size_t i = 0; i < highstride / lowstride; i++)
                    {
                        for(size_t j = 0; j < lowstride; j++)
                        {
                            outbuf[o * lowstride + j] = max(outbuf[o * lowstride + j],
                                inbuf[o * highstride + i * lowstride + j]);
                        }
                    }
                }
            }

            auto axes = op.attributes["axes"].get!(size_t[]);
            auto shape = op.deps[0].shape.dup;

            auto inbuf = inputs[0].as!T;
            T[] outbuf;

            foreach(axis; axes)
            {
                //auto axis = axes[0];
                auto newvol = shape.fold!((a, b) => a * b)(size_t(1)) / shape[axis];
                size_t lowstride;
                
                if(axis == shape.length - 1)
                {
                    lowstride = 1;
                }
                else
                {
                    lowstride = shape[axis + 1 .. $].fold!((a, b) => a * b)(size_t(1));
                }

                size_t highstride = lowstride * shape[axis];

                outbuf = new T[newvol];
                process(inbuf, outbuf, highstride, lowstride);
                inbuf = outbuf;

                shape[axis] = 1;
            }

            output.as!T[] = outbuf[];
        }

        switch(op.outputType.elementType)
        {
            case DataType.float32:
                run!float();
                break;

            case DataType.int32:
                run!int();
                break;

            default:
                throw new Exception("Not implemented.");
        }
    }

    void argminKernel(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        void run(T)()
        {
            auto inbuf = inputs[0].as!T;
            auto outbuf = output.as!int;

            size_t axis = op.attributes["axis"].get!size_t;
            size_t outer = 1;
            size_t inner;
            size_t vol = 1;

            for(size_t i = 0; i < op.deps[0].rank; i++)
            {
                if(i < axis)
                {
                    outer *= op.deps[0].shape[i];
                }
                else if(i > axis)
                {
                    vol *= op.deps[0].shape[i];
                }
                else
                {
                    inner = op.deps[0].shape[i];
                }
            }

            auto vals = new T[vol];

            for(size_t o = 0; o < outer; o++)
            {
                vals[] = T.max;
                
                for(size_t i = 0; i < inner; i++)
                {
                    for(size_t j = 0; j < vol; j++)
                    {
                        if(inbuf[o * vol * inner + i * vol + j] < vals[j])
                        {
                            vals[j] = inbuf[o * vol * inner + i * vol + j];
                            outbuf[o * vol + j] = cast(int)i;
                        }
                    }
                }
            }
        }

        switch(op.deps[0].outputType.elementType)
        {
            case DataType.float32:
                run!float();
                break;

            case DataType.int32:
                run!int();
                break;

            default:
                throw new Exception("Not implemented.");
        }
    }

    immutable string[] arith = ["add", "sub", "mul", "div"];
    immutable string[] comp = ["lt", "lte", "gt", "gte", "eq", "neq"];
    immutable string[] binfunc = ["max", "min", "pow", "atan2"];
    immutable string[] unfunc = ["neg", "abs", "sgn", "exp", "log", "sqrt", "sin", "cos", "tan", "asin", "acos",
        "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"];
    
    string generateRegistrations()
    {
        return chain(arith, comp, binfunc, unfunc)
              .map!(x => "registerCPUKernel(\"" ~ x ~ "\", new CPUKernelDelegate(toDelegate(&" ~ x ~ "Kernel)));")
              .joiner("\n")
              .to!string;
    }

    string generateKernels()
    {
        string[string] opsymbol = ["add": "+", "sub": "-", "mul": "*", "div": "/", "lt": "<", "lte": "<=",
                                         "gt": ">", "gte": ">=", "eq": "==", "neq": "!=", "neg": "-",
                                         "exp": "expCast", "sqrt": "sqrtCast"];

        string[string] types = ["float": "float32", "int": "int32"];

        string[] kernelStrings;

        //This is used for generating a kernel for a specific operation and type combination
        string generateSingleKernel(string op, string dtype, string expr)
        {
            return
                "void " ~ op ~ "Kernel_" ~ dtype ~ "(Operation op, const(Buffer)[] inputs, Buffer output)
                {
                    auto ins = inputs.map!(x => x.as!" ~ dtype ~ ").array();
                    auto outs = output.as!" ~ dtype ~ ";

                    for(size_t i = 0; i < outs.length; i++)
                    {
                        outs[i] = cast(" ~ dtype ~ ")(" ~ expr ~ ");
                    }
                }
                ";
        }

        //Dispatches the arguments to the correct kernel, as determined by the output type of the operation
        string generateTypedKernel(string op, string[string] types)
        {
            string ret =
                "void " ~ op ~ "Kernel(Operation op, const(Buffer)[] inputs, Buffer output)
                {
                    switch(op.outputType.elementType)
                    {
                        ";

            foreach(dtype, vtype; types)
            {
                ret ~= "case DataType." ~ vtype ~ ": " ~ op ~ "Kernel_" ~ dtype ~ "(op, inputs, output); break;\n";
            }

            ret ~= "default: throw new Exception(\"Unknown data type\");
                }
            }
            ";

            return ret;
        }

        //Iterate over each type of (binary) operation and generate the kernels
        foreach(op; chain(arith, comp, binfunc))
        {
            string sym = opsymbol.get(op, "");
            string expr;

            if(sym == "")
            {
                expr = op ~ "(ins[0][i], ins[1][i])";
            }
            else
            {
                expr = "ins[0][i] " ~ sym ~ "ins[1][i]";
            }

            auto mux = generateTypedKernel(op, types);
            auto kerns = types
                        .keys
                        .map!(x => generateSingleKernel(op, x, expr))
                        .joiner()
                        .to!string;

            kernelStrings ~= mux;
            kernelStrings ~= kerns;
        }

        //Generates kernels for unary operations
        foreach(op; unfunc)
        {
            string sym = opsymbol.get(op, "");
            string expr;

            if(sym == "")
            {
                expr = op ~ "(cast(float)ins[0][i])";
            }
            else
            {
                expr = sym ~ "(ins[0][i])";
            }

            auto mux = generateTypedKernel(op, types);
            auto kerns = types
                        .keys
                        .map!(x => generateSingleKernel(op, x, expr))
                        .joiner()
                        .to!string;

            kernelStrings ~= mux;
            kernelStrings ~= kerns;
        }

        //Return all the source code we've generated so it can be mixed in
        return kernelStrings.joiner().to!string;
    }
}