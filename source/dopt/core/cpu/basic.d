module dopt.core.cpu.basic;

import dopt.core;

package
{
    void initialize()
    {
        import std.functional : toDelegate;

        registerCPUKernel("slice", new CPUKernelDelegate(toDelegate(&slice)));
        registerCPUKernel("pad", new CPUKernelDelegate(toDelegate(&pad)));
        registerCPUKernel("transpose", new CPUKernelDelegate(toDelegate(&transpose)));
        registerCPUKernel("repeat", new CPUKernelDelegate(toDelegate(&repeat)));
    }
}

private
{
    void slice(Operation op, const(Buffer)[] inputs, Buffer output)
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
        auto offset = op.attributes["start"].get!(size_t[]);

        if(inShape.length > 0)
        {
            inVol /= inShape[0];
            outVol /= outShape[0];
        }

        sliceImpl(inputs[0].as!ubyte, inShape, inVol, output.as!ubyte, outShape, outVol, offset);
    }

    void pad(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        size_t size = 4;

        void padImpl(const(ubyte[]) input, size_t[] inShape, size_t inVol,
                     ubyte[] output, size_t[] outShape, size_t outVol, size_t[] offset)
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
        auto offset = op.attributes["before"].get!(size_t[]);

        if(inShape.length > 0)
        {
            inVol /= inShape[0];
            outVol /= outShape[0];
        }

        padImpl(inputs[0].as!ubyte, inShape, inVol, output.as!ubyte, outShape, outVol, offset);
    }

    void transpose(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        import std.exception : enforce;
        enforce(op.outputType.rank <= 2, "transpose is only implemented for rank <= 2");

        //Check whether we actually need to reorder them..
        auto order = op
                    .attributes["order"]
                    .get!(size_t[]);

        if(order == [0, 1])
        {
            return;
        }

        if(op.outputType.rank < 2)
        {
            output.as!ubyte[] = inputs[0].as!ubyte[];
        }
        else
        {
            auto inBuf = inputs[0].as!ubyte;
            auto outBuf = output.as!ubyte;
            size_t size = outBuf.length / op.outputType.volume;
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

    void repeat(Operation op, const(Buffer)[] inputs, Buffer output)
    {
        void run(T)()
        {
            void process(const(T)[] inbuf, T[] outbuf, size_t reps, size_t vol)
            {
                import std.array : array;
                import std.range : iota;
                import std.parallelism : parallel;

                //for(size_t i = 0; i < inbuf.length; i += vol)
                foreach(i; iota(0, inbuf.length, vol).array().parallel)
                {
                    for(size_t o = i * reps; o < (i + vol) * reps; o += vol)
                    {
                        outbuf[o .. o + vol] = inbuf[i .. i + vol];
                    }
                }
            }

            //Iterate over each axis, from smallest stride to largest stride
            size_t vol = 1;
            auto inbuf = inputs[0].as!T;
            T[] outbuf;

            foreach_reverse(i, a; op.attributes["repetitions"].get!(size_t[]))
            {
                vol *= op.deps[0].shape[i];
                outbuf = new T[inbuf.length * a];
                process(inbuf, outbuf, a, vol);
                vol *= a;
                inbuf = outbuf;
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
}