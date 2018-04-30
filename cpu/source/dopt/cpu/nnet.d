module dopt.cpu.nnet;

import dopt.core;
import dopt.cpu;

package
{
    void initialize()
    {
        import std.functional : toDelegate;

        registerCPUKernel("convolution", new CPUKernelDelegate(toDelegate(&convolution)));
        registerCPUKernel("maxpool", new CPUKernelDelegate(toDelegate(&maxpool)));
        registerCPUKernel("softmax", new CPUKernelDelegate(toDelegate(&softmax)));
    }
}

private
{
    void convolution(Operation op, const(void[])[] inputs, void[] output)
    {
        size_t[] inDims = op.deps[0].shape[2 .. $];
        size_t[] outDims = op.shape[2 .. $];
        size_t[] kernDims = op.deps[1].shape[2 .. $];
        size_t[] padding = op.attributes["padding"].get!(size_t[]);
        size_t[] stride = op.attributes["stride"].get!(size_t[]);
        size_t numOutpus = op.deps[1].shape[0];
        size_t numInputs = op.deps[1].shape[1];
        size_t batchSize = op.shape[0];
        size_t inVol = inDims[0] * inDims[1];
        size_t outVol = outDims[0] * outDims[1];
        size_t kernVol = kernDims[0] * kernDims[1];

        void conv2d(const(float)[] inimg, const(float)[] kern, float[] outimg)
        {
            size_t outidx = 0;

            for(size_t y = 0; y < outDims[0]; y++)
            {
                for(size_t x = 0; x < outDims[1]; x++)
                {
                    float outval = 0;

                    for(size_t j = 0; j < kernDims[0]; j++)
                    {
                        size_t jprime = kernDims[0] - j - 1;

                        for(size_t i = 0; i < kernDims[1]; i++)
                        {
                            size_t iprime = kernDims[1] - i - 1;
                            ptrdiff_t iny = y * stride[0] - padding[0] + j;
                            ptrdiff_t inx = x * stride[1] - padding[1] + i;

                            if(0 <= iny && iny < outDims[0] && 0 <= inx && inx < outDims[1])
                            {
                                outval += kern[jprime * kernDims[1] + iprime] * inimg[iny * inDims[1] + inx];
                            }
                        }
                    }

                    outimg[outidx] += outval;
                    outidx++;
                }
            }
        }

        auto inbuf = cast(const(float[]))inputs[0];
        auto kernbuf = cast(const(float[]))inputs[1];
        auto outbuf = cast(float[])output;

        for(size_t b = 0; b < batchSize; b++)
        {
            for(size_t o = 0; o < numOutpus; o++)
            {
                float[] outimg = outbuf[(b * numOutpus + o) * outVol .. (b * numOutpus + o + 1) * outVol];
                outimg[] = 0;

                for(size_t i = 0; i < numInputs; i++)
                {
                    const(float)[] inimg = inbuf[(b * numInputs + i) * inVol .. (b * numInputs + i + 1) * inVol];
                    const(float)[] kern = kernbuf[(o * numInputs + i) * kernVol .. (o * numInputs + i + 1) * kernVol];

                    conv2d(inimg, kern, outimg);
                }
            }
        }
    }

    void maxpool(Operation op, const(void[])[] inputs, void[] output)
    {
        size_t[] poolDims = op.attributes["dims"].get!(size_t[]);
        size_t[] inDims = op.deps[0].shape[2 .. $];
        size_t[] outDims = op.shape[2 .. $];
        size_t numMaps = op.shape[0] * op.shape[1];
        size_t inVol = inDims[0] * inDims[1];
        size_t outVol = outDims[0] * outDims[1];

        void pool(const(float)[] inimg, float[] outimg)
        {
            for(size_t y = 0; y < outDims[0]; y++)
            {
                for(size_t x = 0; x < outDims[1]; x++)
                {
                    float maxval = -float.max;

                    for(size_t j = 0; j < poolDims[0]; j++)
                    {
                        for(size_t i = 0; i < poolDims[1]; i++)
                        {
                            import std.algorithm : max;
                            maxval = max(maxval, inimg[(y * poolDims[0] + j) * inDims[1] + x * poolDims[1] + i]);
                        }
                    }

                    outimg[y * outDims[1] + x] = maxval;
                }
            }
        }

        float[] outbuf = cast(float[])output;
        const(float)[] inbuf = cast(const(float)[])inputs[0];

        for(size_t i = 0; i < numMaps; i++)
        {
            pool(inbuf[i * inVol .. (i + 1) * inVol], outbuf[i * outVol .. (i + 1) * outVol]);
        }
    }

    void softmax(Operation op, const(void[])[] inputs, void[] output)
    {
        const(float)[] inbuf = cast(const(float[]))inputs[0];
        float[] outbuf = cast(float[])output;

        size_t elvol = op.volume / (op.shape[0] * op.shape[1]);

        for(size_t b = 0; b < op.shape[0]; b++)
        {
            for(size_t i = 0; i < elvol; i++)
            {
                float m = -float.max;

                for(size_t o = 0; o < op.shape[1]; o++)
                {
                    import std.algorithm : max;

                    m = max(m, inbuf[b * op.shape[1] * elvol + o * elvol + i]);
                }

                float s = 0;

                for(size_t o = 0; o < op.shape[1]; o++)
                {
                    import std.math : exp;

                    float pot = exp(inbuf[b * op.shape[1] * elvol + o * elvol + i] - m);
                    s += pot;
                    outbuf[b * op.shape[1] * elvol + o * elvol + i] = pot;
                }

                for(size_t o = 0; o < op.shape[1]; o++)
                {
                    outbuf[b * op.shape[1] * elvol + o * elvol + i] /= s;
                }
            }
        }
    }
}