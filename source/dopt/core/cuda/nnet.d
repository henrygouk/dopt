module dopt.core.cuda.nnet;

import std.algorithm;
import std.array;
import std.functional;

import dopt.core.cuda;
import dopt.core.ops;

import derelict.cudnn;

package
{
    void initialize()
    {
        DerelictCuDNN.load();
        
        registerCUDAKernel("convolution", toDelegate(&cudaKernelCtr!ConvolutionForward));
        registerCUDAKernel("convolutionFeaturesGrad", toDelegate(&cudaKernelCtr!ConvolutionFeaturesGrad));
        registerCUDAKernel("convolutionFiltersGrad", toDelegate(&cudaKernelCtr!ConvolutionFiltersGrad));
        registerCUDAKernel("maxpool", toDelegate(&cudaKernelCtr!MaxpoolForward));
        registerCUDAKernel("maxpoolGrad", toDelegate(&cudaKernelCtr!MaxpoolGrad));
        registerCUDAKernel("softmax", toDelegate(&cudaKernelCtr!Softmax));
        registerCUDAKernel("softmaxGrad", toDelegate(&cudaKernelCtr!SoftmaxGrad));

        cudnnCreate(&handle);
    }
}

private
{
    cudnnHandle_t handle;

    void cudnnCheck(cudnnStatus_t status, string mod = __MODULE__, size_t line = __LINE__)
    {
        import std.conv : to;
        import std.exception : enforce;
        enforce(status == CUDNN_STATUS_SUCCESS, mod ~ "(" ~ line.to!string ~ "): Failed to execute cuDNN function." ~
            " Error code: " ~ status.to!string);
    }

    CUDAKernel cudaKernelCtr(K)(Operation op)
    {
        return new K(op);
    }

    class ConvolutionBase : CUDAKernel
    {
        this(Operation op, int[] inShape, int[] filterShape, int[] outShape)
        {
            mOp = op;

            cudnnCreateTensorDescriptor(&xDesc).cudnnCheck();
			cudnnCreateFilterDescriptor(&wDesc).cudnnCheck();
			cudnnCreateConvolutionDescriptor(&convDesc).cudnnCheck();
			cudnnCreateTensorDescriptor(&yDesc).cudnnCheck();

            cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inShape[0], inShape[1], inShape[2],
                inShape[3]).cudnnCheck();
            cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterShape[0], filterShape[1],
                filterShape[2], filterShape[3]).cudnnCheck();
            cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CONVOLUTION).cudnnCheck();
            cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outShape[0], outShape[1],
                outShape[2], outShape[3]).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyFilterDescriptor(wDesc).cudnnCheck();
			cudnnDestroyTensorDescriptor(yDesc).cudnnCheck();
			cudnnDestroyConvolutionDescriptor(convDesc).cudnnCheck();
			cudnnDestroyTensorDescriptor(xDesc).cudnnCheck();
        }

        abstract void execute(const(CUDABuffer)[] inputs, CUDABuffer output);

        Operation mOp;
        cudnnTensorDescriptor_t xDesc;
		cudnnFilterDescriptor_t wDesc;
		cudnnTensorDescriptor_t bDesc;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnTensorDescriptor_t yDesc;
    }

    class ConvolutionForward : ConvolutionBase
    {
        this(Operation op)
        {
            auto inShape = op.deps[0].outputType.shape.map!(x => cast(int)x).array();
            auto filterShape = op.deps[1].outputType.shape.map!(x => cast(int)x).array();
            auto outShape = op.outputType.shape.map!(x => cast(int)x).array();

            super(op, inShape, filterShape, outShape);
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto x = cast(void *)inputs[0].ptr;
            auto w = cast(void *)inputs[1].ptr;
            auto y = cast(void *)output.ptr;
            float alpha = 1;
            float beta = 0;

            cudnnConvolutionForward(handle, &alpha, xDesc, x, wDesc, w, convDesc, 0, null, 0, &beta, yDesc, y)
            .cudnnCheck();
        }
    }

    class ConvolutionFeaturesGrad : ConvolutionBase
    {
        this(Operation op)
        {
            auto inShape = op.shape.map!(x => cast(int)x).array();
            auto filterShape = op.deps[1].shape.map!(x => cast(int)x).array();
            auto outShape = op.deps[0].shape.map!(x => cast(int)x).array();

            super(op, inShape, filterShape, outShape);
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto w = cast(void *)inputs[1].ptr;
            auto dy = cast(void *)inputs[0].ptr;
            auto dx = cast(void *)output.ptr;
            float alpha = 1;
            float beta = 0;

            cudnnConvolutionBackwardData(handle, &alpha, wDesc, w, yDesc, dy, convDesc, 0, null, 0, &beta, xDesc, dx)
            .cudnnCheck();
        }
    }

    class ConvolutionFiltersGrad : ConvolutionBase
    {
        this(Operation op)
        {
            auto inShape = op.deps[1].outputType.shape.map!(x => cast(int)x).array();
            auto filterShape = op.outputType.shape.map!(x => cast(int)x).array();
            auto outShape = op.deps[0].outputType.shape.map!(x => cast(int)x).array();

            super(op, inShape, filterShape, outShape);
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto x = cast(void *)inputs[1].ptr;
            auto dy = cast(void *)inputs[0].ptr;
            auto dw = cast(void *)output.ptr;
            float alpha = 1;
            float beta = 0;

            cudnnConvolutionBackwardFilter(handle, &alpha, xDesc, x, yDesc, dy, convDesc, 0, null, 0, &beta, wDesc,
                dw).cudnnCheck();
        }
    }

    class MaxpoolBase : CUDAKernel
    {
        this(Operation op, int[] inShape, int[]outShape)
        {
            auto dims = op.attributes["dims"].get!(size_t[]);
            auto poolShape = dims.map!(x => cast(int)x).array();
            auto poolStride = poolShape.dup;

            cudnnCreatePoolingDescriptor(&poolingDesc).cudnnCheck();
			cudnnSetPooling2dDescriptor(poolingDesc, CUDNN_POOLING_MAX, 1, cast(int)poolShape[0],
                cast(int)poolShape[1], 0, 0, cast(int)poolStride[0], cast(int)poolStride[1]).cudnnCheck();

			cudnnCreateTensorDescriptor(&xDesc).cudnnCheck();
			cudnnCreateTensorDescriptor(&yDesc).cudnnCheck();
			cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inShape[0], inShape[1], inShape[2],
                inShape[3]).cudnnCheck();
			cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, outShape[0], outShape[1],
                outShape[2], outShape[3]).cudnnCheck();

        }

        ~this()
		{
			cudnnDestroyPoolingDescriptor(poolingDesc).cudnnCheck();
			cudnnDestroyTensorDescriptor(xDesc).cudnnCheck();
			cudnnDestroyTensorDescriptor(yDesc).cudnnCheck();
		}

        abstract void execute(const(CUDABuffer)[] inputs, CUDABuffer output);

        cudnnPoolingDescriptor_t poolingDesc;
		cudnnTensorDescriptor_t xDesc;
		cudnnTensorDescriptor_t yDesc;
    }

    class MaxpoolForward : MaxpoolBase
    {
        this(Operation op)
        {
            auto inShape = op.deps[0].outputType.shape.map!(x => cast(int)x).array();
			auto outShape = op.outputType.shape.map!(x => cast(int)x).array();

            super(op, inShape, outShape);
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto x = cast(void *)inputs[0].ptr;
			auto y = cast(void *)output.ptr;
			float alpha = 1;
			float beta = 0;

			cudnnPoolingForward(handle, poolingDesc, &alpha, xDesc, x, &beta, yDesc, y).cudnnCheck();
        }
    }

    class MaxpoolGrad : MaxpoolBase
    {
        this(Operation op)
        {
            auto inShape = op.deps[2].outputType.shape.map!(x => cast(int)x).array();
			auto outShape = op.deps[1].outputType.shape.map!(x => cast(int)x).array();

            super(op, inShape, outShape);
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto dx = cast(void *)output.ptr;
			auto dy = cast(void *)inputs[0].ptr;
			auto y = cast(void *)inputs[1].ptr;
			auto x = cast(void *)inputs[2].ptr;
			float alpha = 1;
			float beta = 0;

			cudnnPoolingBackward(handle, poolingDesc, &alpha, yDesc, y, yDesc, dy, xDesc, x, &beta, xDesc, dx)
            .cudnnCheck();
        }
    }

    class Softmax : CUDAKernel
    {
        this(Operation op)
        {
            auto shape = op.shape.map!(x => cast(int)x).array();
            auto vol = 1;
            
            for(size_t i = 2; i < shape.length; i++)
            {
                vol *= shape[i];
            }

			cudnnCreateTensorDescriptor(&desc).cudnnCheck();
			cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], vol, 1)
            .cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(desc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            float alpha = 1.0;
			float beta = 0.0;
			auto x = cast(void *)inputs[0].ptr;
			auto y = cast(void *)output.ptr;

			cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc, x, &beta,
                desc, y).cudnnCheck();
        }

        cudnnTensorDescriptor_t desc;
    }

    class SoftmaxGrad : CUDAKernel
    {
        this(Operation op)
        {
            auto shape = op.shape.map!(x => cast(int)x).array();

			cudnnCreateTensorDescriptor(&desc).cudnnCheck();
			cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1],
                reduce!"a * b"(1, shape[2 .. $]), 1).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(desc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            float alpha = 1.0;
			float beta = 0.0;
			auto dy = cast(void *)inputs[0].ptr;
			auto y = cast(void *)inputs[1].ptr;
			auto dx = cast(void *)output.ptr;

			cudnnSoftmaxBackward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc, y, desc, dy,
                &beta, desc, dx).cudnnCheck();
        }

        cudnnTensorDescriptor_t desc;
    }
}