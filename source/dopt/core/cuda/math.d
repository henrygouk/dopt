module dopt.core.cuda.math;

import std.algorithm;
import std.conv;
import std.functional;
import std.math;
import std.range;

import dopt.core.cuda;
import dopt.core.cuda.nvrtc;
import dopt.core.ops;
import dopt.core.types;

import derelict.cuda;

package
{
    extern(C)
    {
        enum
        {
            CUBLAS_STATUS_SUCCESS         =0,
            CUBLAS_STATUS_NOT_INITIALIZED =1,
            CUBLAS_STATUS_ALLOC_FAILED    =3,
            CUBLAS_STATUS_INVALID_VALUE   =7,
            CUBLAS_STATUS_ARCH_MISMATCH   =8,
            CUBLAS_STATUS_MAPPING_ERROR   =11,
            CUBLAS_STATUS_EXECUTION_FAILED=13,
            CUBLAS_STATUS_INTERNAL_ERROR  =14,
            CUBLAS_STATUS_NOT_SUPPORTED   =15,
            CUBLAS_STATUS_LICENSE_ERROR   =16
        }

        alias cublasStatus_t = int;

        enum
        {
            CUBLAS_OP_N=0,  
            CUBLAS_OP_T=1,  
            CUBLAS_OP_C=2  
        }

        alias cublasOperation_t = int;

        enum
        { 
            CUBLAS_POINTER_MODE_HOST   = 0,  
            CUBLAS_POINTER_MODE_DEVICE = 1        
        }

        alias cublasPointerMode_t = int;

        struct cublasContext;
        alias cublasHandle_t = cublasContext *;

        cublasStatus_t function(cublasHandle_t *handle) cublasCreate_v2;
        cublasStatus_t function(cublasHandle_t handle, cublasPointerMode_t mode) cublasSetPointerMode_v2;

        cublasStatus_t function(cublasHandle_t handle, int n, const float *x, int incx, const float *y, int incy, float *result) cublasSdot_v2;
        cublasStatus_t function(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc) cublasSgemm_v2;
        cublasStatus_t function(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, const float *alpha, const float *A, int lda, const float *beta, const float *B, int ldb, float *C, int ldc) cublasSgeam;
    }

    void initialize()
    {
        //TODO: make a DerelictCuBLAS library
        import core.sys.posix.dlfcn;
		import std.string;
		fh = dlopen("libcublas.so".toStringz, RTLD_LAZY);
		cublasCreate_v2 = cast(typeof(cublasCreate_v2))dlsym(fh, "cublasCreate_v2");
		cublasSetPointerMode_v2 = cast(typeof(cublasSetPointerMode_v2))dlsym(fh, "cublasSetPointerMode_v2");
		cublasSdot_v2 = cast(typeof(cublasSdot_v2))dlsym(fh, "cublasSdot_v2");
		cublasSgemm_v2 = cast(typeof(cublasSgemm_v2))dlsym(fh, "cublasSgemm_v2");
		cublasSgeam = cast(typeof(cublasSgeam))dlsym(fh, "cublasSgeam");

        cublasCreate_v2(&mCuBLASHandle);

        mixin(generateRegistrations());
        registerCUDAKernel("matmul", toDelegate(&matmulKernelCtr));
        registerCUDAKernel("sum", toDelegate(&sumKernelCtr));
    }

    static ~this()
	{
		import core.sys.posix.dlfcn;
		dlclose(fh);
	}

    cublasHandle_t mCuBLASHandle;
}

private
{
    void *fh;

    immutable string[] arith = ["add", "sub", "mul", "div"];
    immutable string[] comp = ["lt", "lte", "gt", "gte", "eq", "neq"];
    immutable string[] binfunc = ["max", "min", "pow"];
    immutable string[] unfunc = ["neg", "abs", "sgn", "exp", "log", "sqrt"];
    
    string generateRegistrations()
    {
        return chain(arith, comp, binfunc, unfunc)
              .map!(x => "registerCUDAKernel(\"" ~ x ~ "\", toDelegate(&" ~ x ~ "KernelCtr));")
              .joiner("\n")
              .to!string;
    }

    //Arithmetic operations
    alias addKernelCtr = cudaKernelCtr!("+");
    alias subKernelCtr = cudaKernelCtr!("-");
    alias mulKernelCtr = cudaKernelCtr!("*");
    alias divKernelCtr = cudaKernelCtr!("/");
    
    //Comparison operations
    alias ltKernelCtr = cudaKernelCtr!("<");
    alias lteKernelCtr = cudaKernelCtr!("<=");
    alias gtKernelCtr = cudaKernelCtr!(">");
    alias gteKernelCtr = cudaKernelCtr!(">=");
    alias eqKernelCtr = cudaKernelCtr!("==");
    alias neqKernelCtr = cudaKernelCtr!("!=");

    //Binary functions
    alias maxKernelCtr = cudaKernelCtr!("max", 2, false);
    alias minKernelCtr = cudaKernelCtr!("min", 2, false);
    alias powKernelCtr = cudaKernelCtr!("pow", 2, false);

    //Unary functions
    alias negKernelCtr = cudaKernelCtr!("-", 1, false);
    alias absKernelCtr = cudaKernelCtr!("abs", 1, false);
    alias sgnKernelCtr = cudaKernelCtr!("sgn", 1, false);
    alias expKernelCtr = cudaKernelCtr!("exp", 1, false);
    alias logKernelCtr = cudaKernelCtr!("log", 1, false);
    alias sqrtKernelCtr = cudaKernelCtr!("sqrt", 1, false);

    CUDAKernel cudaKernelCtr(string opName, int deps = 2, bool infix = true)(const(Operation) op)
    {
        static if(deps == 2)
        {
            enum code = `
                //TODO: change this to iterate over elements with a stride to fully exploit SIMD units
                extern "C" __global__ void pointwiseKernel(size_t n, const T *dep1, const T *dep2, T *output)
                {
                    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

                    if(i < n)
                    {
                        output[i] = ` ~ (infix ? "dep1[i] " ~ opName ~ " dep2[i]" : opName ~ "(dep1[i], dep2[i])") ~ `;
                    }
                }
            `;
        }
        else static if(deps == 1)
        {
            enum code = `
                T __device__ sgn(T a)
                {
                    return (T)((T(0) < a) - (a < T(0)));
                }

                //TODO: change this to iterate over elements with a stride to fully exploit SIMD units
                extern "C" __global__ void pointwiseKernel(size_t n, const T *dep1, T *output)
                {
                    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

                    if(i < n)
                    {
                        output[i] = ` ~ opName ~ `(dep1[i]);
                    }
                }
            `;
        }
        else
        {
            assert(0, "Not implemented.");
        }

        return new PointwiseCUDAKernel("typedef " ~ op.outputType.elementType.cudaType() ~ " T;\n" ~ code, op);
    }

    class PointwiseCUDAKernel : CUDAKernel
    {
        this(string code, const(Operation) op)
        {
            mKernel = mKernelCache.get(code, new NVRTCKernel("pointwiseKernel", code));
            mKernelCache[code] = mKernel;
            mOp = op;
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            //Args = [vol, <inputs...>, output]
            void*[] args = new void*[inputs.length + 2];
            CUdeviceptr[] ptrs = new CUdeviceptr[inputs.length + 1];

            size_t n = mOp.outputType.volume;
            args[0] = &n;

            for(size_t i = 0; i < inputs.length; i++)
            {
                ptrs[i] = inputs[i].ptr;
                args[i + 1] = &ptrs[i];
            }

            ptrs[$ - 1] = output.ptr;
            args[$ - 1] = &ptrs[$ - 1];

            uint numThreads = 512;
			uint numBlocks = (cast(uint)n + numThreads) / numThreads;
            mKernel.execute(numBlocks, numThreads, args);
        }

        static NVRTCKernel[string] mKernelCache;
        NVRTCKernel mKernel;
        const(Operation) mOp;
    }

    CUDAKernel matmulKernelCtr(const(Operation) op)
    {
        return new MatmulKernel(op);
    }

    class MatmulKernel : CUDAKernel
    {
        this(const(Operation) op)
        {
            mOp = op;
            ashape = mOp.deps[0].outputType.shape.map!(x => cast(int)x).array();
			bshape = mOp.deps[1].outputType.shape.map!(x => cast(int)x).array();
			cshape = mOp.outputType.shape.map!(x => cast(int)x).array();
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            if(mOp.outputType.elementType == DataType.float32)
            {
                auto a = cast(float *)inputs[0].ptr;
				auto b = cast(float *)inputs[1].ptr;
				auto c = cast(float *)output.ptr;
				float alpha = 1;
				float beta = 0;

				cublasSgemm_v2(mCuBLASHandle, CUBLAS_OP_N, CUBLAS_OP_N, bshape[1], ashape[0], ashape[1], &alpha, b,
                    bshape[1], a, ashape[1], &beta, c, cshape[1]);
            }
            else
            {
                throw new Exception("Element type not supported.");
            }
        }

        const(Operation) mOp;
        int[] ashape;
		int[] bshape;
		int[] cshape;
    }

    CUDAKernel sumKernelCtr(const(Operation) op)
    {
        return new SumKernel(op);
    }

    class SumKernel : CUDAKernel
    {
        this(const(Operation) op)
        {
            mInput = variable(TensorType(op.deps[0].elementType, op.deps[0].shape));
            mOp = sum(mInput, op.attributes["axes"].get!(const(size_t)[]));
        }

        override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
        {
            auto inbuf = new byte[inputs[0].numBytes];
            inputs[0].get(inbuf);

            import dopt.core.cpu : evaluate;
            auto outbuf = evaluate(mOp, [mInput: Buffer(inbuf)]);

            output.set(outbuf.as!byte);
        }

        const(Operation) mInput;
        const(Operation) mOp;
    }
}
