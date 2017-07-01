module dopt.core.cpu.basic;

import dopt.core;

static this()
{
    import std.functional : toDelegate;

    registerCPUKernel("slice", new CPUKernelDelegate(toDelegate(&slice)));
    registerCPUKernel("pad", new CPUKernelDelegate(toDelegate(&pad)));
    registerCPUKernel("transpose", new CPUKernelDelegate(toDelegate(&transpose)));
    registerCPUKernel("repeat", new CPUKernelDelegate(toDelegate(&repeat)));
}

private
{
    void slice(const(Operation) op, const(void[])[] inputs, void[] output)
    {
        size_t size = 4;

        void sliceImpl(const(ubyte)[] input, in size_t[] inShape, size_t inVol,
                       ubyte[] output, in size_t[] outShape, size_t outVol, in size_t[] offset)
        {
            if(inShape.length == 0)
            {
                output[] = input[];
            }
            else if(inShape.length == 1)
            {
                output[] = input[offset[0] * size .. (offset[0] + outShape[0]) * size];
            }
            else
            {
                for(size_t i = 0; i < outShape[0]; i++)
                {
                    sliceImpl(input[(i + offset[0]) * inVol * size.. (i + offset[0] + 1) * inVol * size],
                                inShape[1 .. $],
                                inVol / inShape[1],
                                output[i * outVol * size .. (i + 1) * outVol * size],
                                outShape[1 .. $],
                                outVol / outShape[1],
                                offset[1 .. $]);
                }
            }
        }

        auto inShape = op.deps[0].outputType.shape;
        auto outShape = op.outputType.shape;
        size_t inVol = op.deps[0].outputType.volume;
        size_t outVol = op.outputType.volume;
        auto offset = op.attributes["start"].get!(const(size_t)[]);

        if(inShape.length > 0)
        {
            inVol /= inShape[0];
            outVol /= outShape[0];
        }

        sliceImpl(cast(const(ubyte)[])inputs[0], inShape, inVol, cast(ubyte[])output, outShape, outVol, offset);
    }

    void pad(const(Operation) op, const(void[])[] inputs, void[] output)
    {
        size_t size = 4;

        void padImpl(const(ubyte[]) input, const(size_t)[] inShape, size_t inVol,
                     ubyte[] output, const(size_t)[] outShape, size_t outVol, const(size_t)[] offset)
        {
            if(inShape.length == 0)
            {
                output[] = input[];
            }
            else if(inShape.length == 1)
            {
                output[0 .. offset[0] * size] = 0;
                output[offset[0] * size .. (offset[0] + inShape[0]) * size] = input[];
                output[(offset[0] + inShape[0]) * size .. $] = 0;
            }
            else
            {
                output[0 .. offset[0] * outVol * size] = 0;

                for(size_t i = 0; i < inShape[0]; i++)
                {
                    padImpl(input[i * inVol * size.. (i + 1) * inVol * size],
                                inShape[1 .. $],
                                inVol / inShape[1],
                                output[(i + offset[0]) * outVol * size .. (i + offset[0] + 1) * outVol * size],
                                outShape[1 .. $],
                                outVol / outShape[1],
                                offset[1 .. $]);
                }

                output[(offset[0] + inShape[0]) * outVol * size .. $] = 0;
            }
        }

        auto inShape = op.deps[0].outputType.shape;
        auto outShape = op.outputType.shape;
        size_t inVol = op.deps[0].outputType.volume;
        size_t outVol = op.outputType.volume;
        auto offset = op.attributes["before"].get!(const(size_t)[]);

        if(inShape.length > 0)
        {
            inVol /= inShape[0];
            outVol /= outShape[0];
        }

        padImpl(cast(const(ubyte)[])inputs[0], inShape, inVol, cast(ubyte[])output, outShape, outVol, offset);
    }

    void transpose(const(Operation) op, const(void[])[] inputs, void[] output)
    {
        import std.exception : enforce;
        enforce(op.outputType.rank <= 2, "transpose is only implemented for rank <= 2");

        if(op.outputType.rank < 2)
        {
            output[] = inputs[0][];
        }
        else
        {
            auto inBuf = cast(const(ubyte)[])inputs[0];
            auto outBuf = cast(ubyte[])output;
            size_t size = output.length / op.outputType.volume;
            size_t rows = op.outputType.shape[0];
            size_t cols = op.outputType.shape[1];

            for(size_t r = 0; r < rows; r++)
            {
                for(size_t c = 0; c < cols; c++)
                {
                    outBuf[size * (r * cols + c) .. size * (r * cols + c + 1)] =
                        inBuf[size * (c * rows + r) .. size * (c * rows + r + 1)];
                }
            }
        }
    }

    void repeat(const(Operation) op, const(void[])[] inputs, void[] output)
    {
        import std.parallelism : parallel;
        import std.range : chunks;

        auto inBuf = cast(const(ubyte)[])inputs[0];
        auto outBuf = cast(ubyte[])output;

        foreach(c; outBuf.chunks(inBuf.length).parallel)
        {
            c[] = inBuf[];
        }
    }
}