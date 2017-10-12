module dopt.core.cuda.nnet;

import std.algorithm;
import std.array;
import std.functional;

import dopt.core.cuda;
import dopt.core.ops;

import derelict.cuda;
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
        registerCUDAKernel("addBias", toDelegate(&cudaKernelCtr!AddBias));
        registerCUDAKernel("addBiasGrad", toDelegate(&cudaKernelCtr!AddBiasGrad));
        registerCUDAKernel("batchNormTrain", toDelegate(&cudaKernelCtr!BatchNormTrain));
        registerCUDAKernel("batchNormGrad", toDelegate(&cudaKernelCtr!BatchNormGrad));

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

            int padH = 0;
            int padW = 0;
            int strideY = 1;
            int strideX = 1;

            auto padding = op.attributes["padding"].get!(size_t[]);
            padH = cast(int)padding[0];
            padW = cast(int)padding[1];

            auto stride = op.attributes["stride"].get!(size_t[]);
            strideY = cast(int)stride[0];
            strideX = cast(int)stride[1];

            cudnnCreateTensorDescriptor(&xDesc).cudnnCheck();
			cudnnCreateFilterDescriptor(&wDesc).cudnnCheck();
			cudnnCreateConvolutionDescriptor(&convDesc).cudnnCheck();
			cudnnCreateTensorDescriptor(&yDesc).cudnnCheck();

            cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, inShape[0], inShape[1], inShape[2],
                inShape[3]).cudnnCheck();
            cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, filterShape[0], filterShape[1],
                filterShape[2], filterShape[3]).cudnnCheck();
            cudnnSetConvolution2dDescriptor(convDesc, padH, padW, strideY, strideX, 1, 1, CUDNN_CONVOLUTION).cudnnCheck();
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

            cuCtxSynchronize();
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

            cuCtxSynchronize();
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

            cuCtxSynchronize();
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

            cuCtxSynchronize();
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

            cuCtxSynchronize();
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
            
            cuCtxSynchronize();
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

            cuCtxSynchronize();
        }

        cudnnTensorDescriptor_t desc;
    }

    class AddBias : CUDAKernel
    {
        this(Operation op)
        {
            auto shape = op.shape.map!(x => cast(int)x).array();

			cudnnCreateTensorDescriptor(&cDesc).cudnnCheck();
			cudnnSetTensor4dDescriptor(cDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1],
                reduce!"a * b"(1, shape[2 .. $]), 1).cudnnCheck();
            
            cudnnCreateTensorDescriptor(&aDesc).cudnnCheck();
            cudnnSetTensor4dDescriptor(aDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, shape[1], 1, 1).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(cDesc).cudnnCheck();
            cudnnDestroyTensorDescriptor(aDesc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            cuMemcpy(output.ptr, inputs[0].ptr, output.numBytes);

            float alpha = 1;
            float beta = 1;

            cudnnAddTensor(handle, &alpha, aDesc, cast(void *)inputs[1].ptr, &beta, cDesc, cast(void *)output.ptr);
        }

        cudnnTensorDescriptor_t cDesc;
        cudnnTensorDescriptor_t aDesc;
    }

    class AddBiasGrad : CUDAKernel
    {
        this(Operation op)
        {
            auto shape = op.deps[0].shape.map!(x => cast(int)x).array();

			cudnnCreateTensorDescriptor(&dyDesc).cudnnCheck();
			cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1],
                reduce!"a * b"(1, shape[2 .. $]), 1).cudnnCheck();
            
            cudnnCreateTensorDescriptor(&dbDesc).cudnnCheck();
            cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, shape[1], 1, 1).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(dyDesc).cudnnCheck();
            cudnnDestroyTensorDescriptor(dbDesc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            float alpha = 1.0f;
            float beta = 1.0f;

            cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, cast(void *)inputs[0].ptr, &beta, dbDesc,
                cast(void *)output.ptr);
        }

        cudnnTensorDescriptor_t dyDesc;
        cudnnTensorDescriptor_t dbDesc;
    }

    class BatchNormTrain : CUDAKernel
    {
        this(Operation op)
        {
            if(op.rank == 2)
            {
                mode = CUDNN_BATCHNORM_PER_ACTIVATION;
            }
            else
            {
                mode = CUDNN_BATCHNORM_SPATIAL;
            }

            auto shape = op.deps[0].shape.map!(x => cast(int)x).array();

            cudnnCreateTensorDescriptor(&xDesc).cudnnCheck();
            cudnnCreateTensorDescriptor(&bnDesc).cudnnCheck();

            cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2],
                shape[3]).cudnnCheck();
            cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, mode).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(xDesc).cudnnCheck();
            cudnnDestroyTensorDescriptor(bnDesc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            float alpha = 1.0f;
            float beta = 0.0f;

            cudnnBatchNormalizationForwardTraining(handle, mode, &alpha, &beta, xDesc,
                cast(void *)inputs[0].ptr, xDesc, cast(void *)output.ptr, bnDesc, cast(void *)inputs[1].ptr,
                cast(void *)inputs[2].ptr, 0, null, null, 1e-5f, null, null).cudnnCheck();
        }

        cudnnBatchNormMode_t mode;
        cudnnTensorDescriptor_t xDesc;
        cudnnTensorDescriptor_t bnDesc;
    }

    class BatchNormGrad : CUDAKernel
    {
        this(Operation op)
        {
            if(op.deps[1].rank == 2)
            {
                mode = CUDNN_BATCHNORM_PER_ACTIVATION;
            }
            else
            {
                mode = CUDNN_BATCHNORM_SPATIAL;
            }

            auto shape = op.deps[1].shape.map!(x => cast(int)x).array();

            cudnnCreateTensorDescriptor(&xDesc).cudnnCheck();
            cudnnCreateTensorDescriptor(&bnDesc).cudnnCheck();

            cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, shape[0], shape[1], shape[2],
                shape[3]).cudnnCheck();
            cudnnDeriveBNTensorDescriptor(bnDesc, xDesc, mode).cudnnCheck();
        }

        ~this()
        {
            cudnnDestroyTensorDescriptor(xDesc).cudnnCheck();
            cudnnDestroyTensorDescriptor(bnDesc).cudnnCheck();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            float alpha = 1.0f;
            float beta = 0.0f;

            void *dx = cast(void *)(output.ptr);
            void *dscale = cast(void *)(output.ptr + inputs[1].numBytes);
            void *dbias = cast(void *)(output.ptr + inputs[1].numBytes + inputs[2].numBytes);

            cudnnBatchNormalizationBackward(handle, mode, &alpha, &beta, &alpha, &beta, xDesc,
                cast(void *)inputs[1].ptr, xDesc, cast(void *)inputs[0].ptr, xDesc, dx, bnDesc,
                cast(void *)inputs[2].ptr, dscale, dbias, 1e-5f, null, null);
        }

        cudnnBatchNormMode_t mode;
        cudnnTensorDescriptor_t xDesc;
        cudnnTensorDescriptor_t bnDesc;
    }
}