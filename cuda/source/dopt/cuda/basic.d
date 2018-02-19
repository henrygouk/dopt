module dopt.cuda.basic;

import std.functional;

import dopt.cuda;
import dopt.cuda.math;
import dopt.cuda.nvrtc;
import dopt.core.ops;
import dopt.core.types;

import derelict.cuda;

package
{
    void initialize()
    {
        registerCUDAKernel("slice", toDelegate(&cudaKernelCtr!Slice));
        registerCUDAKernel("pad", toDelegate(&cudaKernelCtr!Pad));
        registerCUDAKernel("repeat", toDelegate(&cudaKernelCtr!Repeat));
        registerCUDAKernel("transpose", toDelegate(&cudaKernelCtr!Transpose));
    }
}

private
{
    CUDAKernel cudaKernelCtr(K)(Operation op)
    {
        return new K(op);
    }

    class Slice : CUDAKernel
    {
        this(Operation op)
        {
            mOp = op;
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            size_t size = 4;

            void sliceImpl(const(CUdeviceptr) inputptr, in size_t[] inShape, size_t inVol,
                        CUdeviceptr outputptr, in size_t[] outShape, size_t outVol, in size_t[] offset)
            {
                if(inShape.length == 0)
                {
                    cuMemcpy(outputptr, inputptr, size);
                }
                else if(inShape.length == 1)
                {
                    cuMemcpy(outputptr, inputptr + offset[0] * size, outShape[0] * size);
                }
                else
                {
                    for(size_t i = 0; i < outShape[0]; i++)
                    {
                        sliceImpl(inputptr + (i + offset[0]) * inVol * size,
                                    inShape[1 .. $],
                                    inVol / inShape[1],
                                    outputptr + i * outVol * size,
                                    outShape[1 .. $],
                                    outVol / outShape[1],
                                    offset[1 .. $]);
                    }
                }
            }

            auto inShape = mOp.deps[0].outputType.shape;
            auto outShape = mOp.outputType.shape;
            size_t inVol = mOp.deps[0].outputType.volume;
            size_t outVol = mOp.outputType.volume;
            auto offset = mOp.attributes["start"].get!(size_t[]);

            if(inShape.length > 0)
            {
                inVol /= inShape[0];
                outVol /= outShape[0];
            }

            sliceImpl(inputs[0].ptr, inShape, inVol, output.ptr, outShape, outVol, offset);
        }

        Operation mOp;
    }

    class Pad : CUDAKernel
    {
        this(Operation op)
        {
            mOp = op;
        }

        void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            size_t size = 4;

            void padImpl(CUdeviceptr inputptr, size_t[] inShape, size_t inVol,
                        CUdeviceptr outputptr, size_t[] outShape, size_t outVol, size_t[] offset)
            {
                if(inShape.length == 0)
                {
                    cuMemcpy(outputptr, inputptr, size);
                }
                else if(inShape.length == 1)
                {
                    cuMemcpy(outputptr + offset[0] * size, inputptr, inShape[0] * size);
                }
                else
                {
                    for(size_t i = 0; i < inShape[0]; i++)
                    {
                        padImpl(inputptr + i * inVol * size,
                                    inShape[1 .. $],
                                    inVol / inShape[1],
                                    outputptr + (i + offset[0]) * outVol * size,
                                    outShape[1 .. $],
                                    outVol / outShape[1],
                                    offset[1 .. $]);
                    }
                }
            }

            auto inShape = mOp.deps[0].outputType.shape;
            auto outShape = mOp.outputType.shape;
            size_t inVol = mOp.deps[0].outputType.volume;
            size_t outVol = mOp.outputType.volume;
            auto offset = mOp.attributes["before"].get!(size_t[]);

            if(inShape.length > 0)
            {
                inVol /= inShape[0];
                outVol /= outShape[0];
            }

            cuMemsetD8(output.ptr, 0, output.numBytes);

            padImpl(inputs[0].ptr, inShape, inVol, output.ptr, outShape, outVol, offset);
        }

        Operation mOp;
    }

    class Repeat : CUDAKernel
    {
        this(Operation op)
        {
            mOp = op;

            if(mRepeatKernel is null)
            {
                mRepeatKernel = new NVRTCKernel("repeatBlocks", `
extern "C" __global__ void repeatBlocks(const char *inbuf, size_t elemSize, size_t len, char *outbuf, size_t reps)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if(i < elemSize * reps * len)
    {
        size_t e = i % elemSize;
        size_t l = i / (elemSize * reps);
        outbuf[i] = inbuf[l * elemSize + e];
    }
}
                `);
            }
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            if(inputs[0].numBytes == output.numBytes)
            {
                cuMemcpy(output.ptr, inputs[0].ptr, output.numBytes);
                return;
            }

            void process(CUDABuffer inbuf, CUDABuffer outbuf, size_t elemSize, size_t len, size_t reps)
            {
                uint n = cast(uint)(elemSize * len * reps);
                uint numThreads = 512;
			    uint numBlocks = (cast(uint)n + numThreads) / numThreads;
                mRepeatKernel.execute(numBlocks, numThreads, inbuf.ptr, elemSize, len, outbuf.ptr, reps);
            }

            //Iterate over each axis, from smallest stride to largest stride
            size_t elemSize = sizeOf(mOp.elementType);
            auto inbuf = cast(CUDABuffer)inputs[0];
            CUDABuffer outbuf;

            foreach_reverse(i, a; mOp.attributes["repetitions"].get!(size_t[]))
            {
                elemSize *= mOp.deps[0].shape[i];

                if(a == 1)
                {
                    continue;
                }
                
                outbuf = CUDABuffer.create(inbuf.numBytes * a);
                
                process(inbuf, outbuf, elemSize, inbuf.numBytes / elemSize, a);

                elemSize *= a;

                if(inbuf.ptr != inputs[0].ptr)
                {
                    CUDABuffer.destroy(inbuf);
                }

                inbuf = outbuf;
            }

            cuMemcpy(output.ptr, outbuf.ptr, output.numBytes);
            CUDABuffer.destroy(outbuf);
        }

        Operation mOp;
        static NVRTCKernel mRepeatKernel;
    }

    class Transpose : CUDAKernel
    {
        this(Operation op)
        {
            mOp = op;
        }

        void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            if(mOp.outputType.elementType == DataType.float32)
            {
                auto a = cast(float *)inputs[0].ptr;
				auto c = cast(float *)output.ptr;
				float alpha = 1;
				float beta = 0;

                auto mShape = mOp.outputType.shape;

				cublasSgeam(mCuBLASHandle, CUBLAS_OP_T, CUBLAS_OP_T, cast(int)mShape[1], cast(int)mShape[0], &alpha, a,
                    cast(int)mShape[0], &beta, a, cast(int)mShape[0], c, cast(int)mShape[1]);
			}
            else
            {
                throw new Exception("Element type not supported.");
            }
        }

        Operation mOp;
    }
}