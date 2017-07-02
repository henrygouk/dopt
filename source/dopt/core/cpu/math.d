module dopt.core.cpu.math;

import dopt.core;

import std.algorithm;
import std.conv;
import std.functional;
import std.math;
import std.range;

static this()
{
    mixin(generateRegistrations());

    registerCPUKernel("matmul", new CPUKernelDelegate(toDelegate(&matmulKernel)));
}

mixin(generateKernels());

private
{
    void matmulKernel(const(Operation) op, const(Buffer)[] inputs, Buffer output)
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

    immutable string[] arith = ["add", "sub", "mul", "div"];
    immutable string[] comp = ["lt", "lte", "gt", "gte", "eq", "neq"];
    immutable string[] binfunc = ["max", "min", "pow"];
    immutable string[] unfunc = ["neg", "abs"];
    
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
                                         "gt": ">", "gte": ">=", "eq": "==", "neq": "!=", "neg": "-"];

        string[string] types = ["float": "float32", "int": "int32"];

        string[] kernelStrings;

        string generateSingleKernel(string op, string dtype, string expr)
        {
            return
                "void " ~ op ~ "Kernel_" ~ dtype ~ "(const(Operation) op, const(Buffer)[] inputs, Buffer output)
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

        string generateTypedKernel(string op, string[string] types)
        {
            string ret =
                "void " ~ op ~ "Kernel(const(Operation) op, const(Buffer)[] inputs, Buffer output)
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

        foreach(op; unfunc)
        {
            string sym = opsymbol.get(op, "");
            string expr;

            if(sym == "")
            {
                expr = op ~ "(ins[0][i])";
            }
            else
            {
                expr = sym ~ "ins[0][i]";
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

        return kernelStrings.joiner().to!string;
    }
}