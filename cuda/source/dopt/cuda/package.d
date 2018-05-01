/**
    This is the main interface for the dopt CUDA backend.

    The APIs in this module allow users to evaluate operation graphs on GPUs through the use of CUDA. There is also
    functionality to register CUDA implementations of custom operations.

    In future, this module will also have an interface allowing the user to register their own optimisation passes to
    be called when constructing a plan.

    Authors: Henry Gouk
*/
module dopt.cuda;

import std.exception;

import dopt.cuda.basic;
import dopt.cuda.nvrtc;
import dopt.cuda.math;
import dopt.cuda.nnet;
import dopt.cuda.random;
import dopt.core;

import derelict.cuda;

alias CUDAKernelCtr = CUDAKernel delegate(Operation op);

private __gshared
{
    CUdevice mDevice;
    CUcontext mContext;
}

/**
    Registers all the kernels for the CUDA backend
*/
shared static this()
{
    try
    {
        DerelictCUDADriver.load();
        
        //Initialise CUDA and create a context
        cuInit(0);
        cuDeviceGet(&mDevice, 0);
        cuCtxCreate(&mContext, 0, mDevice);

        //Initialize submodules
        dopt.cuda.basic.initialize();
        dopt.cuda.nvrtc.initialize();
        dopt.cuda.math.initialize();
        dopt.cuda.nnet.initialize();
        dopt.cuda.random.initialize();

        import std.functional : toDelegate;
        defaultEvaluator = toDelegate(&evaluateCUDA);
        defaultCompiler = (Operation[] ops) { return new CUDAPlan(ops); };
        defaultVarAllocator = (size_t numBytes) { return CUDABuffer.create(numBytes); };
    }
    catch(Exception e)
    {
        //TODO: probably log something here
    }
}

/**
    Provides a common interface for CUDA kernels.
*/
interface CUDAKernel
{
    /**
        Runs the kernel with the given inputs and outputs.

        Params:
            inputs = An array of CUDABuffer objects, each corresponding to one of the dependencies of the operation
            used to construct this kernel.
            output = The destination buffer.
    */
    void execute(const(CUDABuffer)[] inputs, CUDABuffer output);
}

private class CUDACPUKernel : CUDAKernel
{
    this(Operation op)
    {
        import std.algorithm : map;
        import std.array : array;

        mDeps = op
               .deps
               .map!(x => variable(x.outputType))
               .array();
        
        mOp = createOperation(op.opType, mDeps, op.attributes);
    }

    void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
    {
        import std.range : zip;
        import dopt.cpu : evaluateCPU;

        foreach(cudaInput, cpuInput; zip(inputs, mDeps))
        {
            cpuInput.value.set(cudaInput);
        }

        DeviceBuffer ret = evaluateCPU([mOp])[0];

        output.set(ret);
    }

    DeviceBuffer[] mInputs;
    Operation[] mDeps;
    Operation mOp;
}

private CUDAKernel cudaCPUCtr(Operation op)
{
    return new CUDACPUKernel(op);
}

/**
    A class that encapsulates the CUDA memory allocation/deallocation process.
*/
class CUDABuffer : DeviceBuffer
{
    public
    {
        /**
            Constructs a CUDABuffer object and allocates memory on the CUDA device.

            Params:
                numBytes = The number of bytes to be allocated on the CUDA device.
        */
        static CUDABuffer create(size_t numBytes)
        {
            import core.memory : GC;

            //Rely on the GC to run some finalisers to free CUDA memory. I know this is bad please help.
            GC.collect();

            CUDABuffer ret = new CUDABuffer();
            ret.mNumBytes = numBytes;
            enforce(cuMemAlloc(&(ret.mPtr), ret.mNumBytes) == CUDA_SUCCESS, "CUDA memory allocation failed");
            enforce(cuMemsetD8(ret.mPtr, 0, ret.mNumBytes) == CUDA_SUCCESS,
                "CUDA default buffer initialisation failed");

            return ret;
        }

        /**
            Releases the CUDA resources used by buf internally.
        */
        static void destroy(CUDABuffer buf)
        {
            enforce(cuMemFree(buf.mPtr) == CUDA_SUCCESS, "Failed to free CUDA device memory.");
        }

        /**
            Copies data from the host to the device.

            Params:
                buf = An array of data to be copied to the device.
        */
        override void set(const void[] buf)
        {
            enforce(buf.length == mNumBytes, "input buffer is the wrong length.");
			enforce(cuMemcpyHtoD(mPtr, buf.ptr, buf.length) == CUDA_SUCCESS, "Failed to set contents of CUDA buffer");
        }

        override void set(const DeviceBuffer buf)
        {
            import dopt.cpu : CPUBuffer;

            enforce(numBytes == buf.numBytes, "Mismatch in buffer size");

            auto cubuf = cast(CUDABuffer)buf;
            auto cpubuf = cast(CPUBuffer)buf;

            if(cubuf !is null)
            {
                cuMemcpyDtoD(mPtr, cubuf.ptr, numBytes);
            }
            else if(cpubuf !is null)
            {
                cuMemcpyHtoD(mPtr, cpubuf.raw.ptr, numBytes);
            }
            else
            {
                super.set(buf);
            }
        }

        /**
            Copies data from the device to the host.

            Params:
                buf = The buffer that the data from the CUDA device will be written to.
        */
        override void get(void[] buf) const
        {
            enforce(buf.length == mNumBytes, "output buffer is the wrong length.");
			enforce(cuMemcpyDtoH(buf.ptr, mPtr, buf.length) == CUDA_SUCCESS, "Failed to get contents of CUDA buffer");
        }

        /**
            Provides the size of the buffer allocated on the CUDA device.

            Returns:
                The number of bytes allocated on the CUDA device.
        */
        override size_t numBytes() const
        {
            return mNumBytes;
        }

        /**
            Provides the device pointer.

            Returns:
                A CUDA device pointer.
        */
        inout(CUdeviceptr) ptr() inout
        {
            return mPtr;
        }
    }

    private
    {
        size_t mNumBytes;
        CUdeviceptr mPtr;

        this()
        {
            //
        }

        void zero()
        {
            enforce(cuMemsetD8(mPtr, 0, mNumBytes) == CUDA_SUCCESS, "CUDA zero buffer failed");
        }
    }
}

/**
    A Plan stores all the resources (preallocated buffers, custom CUDA kernels) required to evaluate nodes from the
    Operation graph.

    An instance of Plan can be constructed using the $(D compileCUDA) function. The primary use case for a CUDAPlan is when the
    same set of operations are likely to be evaluated more than once. This prevents the dopt CUDA runtime from
    reallocating and optimising the CUDA kernels every time the same set of operations is to be executed.
*/
class CUDAPlan : Plan
{
    public
    {
        long[string] profiler;

        this(Operation[] outputs)
        {
            import std.algorithm : canFind, filter;
            import std.array : array;
            import std.functional : toDelegate;

            super(outputs);

            auto sortedOps = topologicalSort(outputs);

            foreach(o; sortedOps)
            {
                if(o.opType == "variable" || o.opType == "reshape" || o.opType == "constant")
                {
                    continue;
                }
                
                auto k = mKernelCtrs.get(o.opType, toDelegate(&cudaCPUCtr));

                enforce(k !is null, "Could not construct a CUDA kernel for operation of type '" ~ o.opType ~ "'");

                mKernels[o] = k(o);
            }

            mOps = sortedOps.array;

            foreach(o; mOps)
            {
                if(o.opType == "reshape")
                {
                    //This will be overwritten in executeImpl, but we want a slot in the hashmap for it now.
                    mResults[o] = mResults[o.deps[0]];
                }
                else
                {
                    mResults[o] = CUDABuffer.create(o.volume * o.elementType.sizeOf);

                    if(o.opType == "constant")
                    {
                        mResults[o].set(o.value);
                    }
                }
            }

            mResults.rehash();
        }

        ~this()
        {
            cleanup();
        }

        /**
            Releases CUDA resources associated with this plan.
        */
        void cleanup()
        {
            if(mClean)
            {
                return;
            }

            foreach(o; mOps)
            {
                if(o.opType != "reshape")
                {
                    CUDABuffer.destroy(mResults[o]);
                }
            }

            mClean = true;
        }
    }

    protected
    {
        override void executeImpl(DeviceBuffer[Operation] args, DeviceBuffer[] rets)
        {
            import std.datetime.stopwatch : StopWatch;
            StopWatch sw;

            //Make sure all the args are variable assignments. Is this arbitrary?
            foreach(o; args.keys)
            {
                enforce(o.opType == "variable",
                    "All assignments in args must be for Operations with an opType of 'variable'");
            }

            //Iterate through each operation and execute it
            foreach(o; mOps)
            {
                if(o.opType == "variable" || o.opType == "constant")
                {
                    continue;
                }

                //Get the input buffers
                CUDABuffer[] inputs;
                CUDABuffer output = mResults[o];

                foreach(d; o.deps)
                {
                    if(d.opType == "variable")
                    {
                        CUDABuffer cubuf;

                        if(d in args)
                        {
                            cubuf = cast(CUDABuffer)args[d];

                            if(cubuf is null)
                            {
                                cubuf = mResults[d];
                                cubuf.set(args[d]);
                            }
                        }
                        else
                        {
                            cubuf = cast(CUDABuffer)d.value;

                            if(cubuf is null)
                            {
                                cubuf = mResults[d];
                                cubuf.set(d.value);
                            }

                        }

                        inputs ~= cubuf;
                    }
                    else
                    {
                        inputs ~= mResults[d];
                    }
                }

                if(o.opType == "reshape")
                {
                    mResults[o] = inputs[0];
                }
                else
                {
                    //Execute the operation
                    sw.reset();
                    sw.start();
                    mKernels[o].execute(inputs, output);
                    sw.stop();

                    profiler[o.opType] = profiler.get(o.opType, 0) + sw.peek.split.usecs;
                }
            }

            foreach(i, o; mOutputs)
            {
                rets[i].set(mResults[o]);
            }
        }
    }

    private
    {
        Operation[] mOps;
        CUDAKernel[Operation] mKernels;
        CUDABuffer[Operation] mResults;
        bool mClean = false;
    }
}

/**
    Used for performing a one-off evaluation of a set of operations.

    If you are planning to operate the same set of operations multiple times, but with different variables assignments,
    then you should construct a $(D CUDAPlan).

    Params:
        ops = The operations to be evaluated.
        args = A set of optional variable assignments.

    Returns:
        The result of evaluating $(D ops).
*/
DeviceBuffer[] evaluateCUDA(Operation[] ops, DeviceBuffer[Operation] args = null)
{
    auto p = new CUDAPlan(ops);
    
    auto ret = p.execute(args);

    return ret;
}

/**
    A convenience overload that evaluates a single operation and returns a single $(D DeviceBuffer).

    Params:
        op = The operation to be evaluated.
        args = A set of optional variable assignments.

    Returns:
        The result of evaluating $(D op)
*/
DeviceBuffer evaluateCUDA(Operation op, DeviceBuffer[Operation] args = null)
{
    return evaluateCUDA([op], args)[0];
}

/**
    Registers a CUDA kernel constructor for a given operation type.

    Params:
        opName = The type of operation this kernel constructor caters to.
        kernelCtr = The constructor that should be associated with operations with the type $(D opType).
*/
void registerCUDAKernel(string opName, CUDAKernelCtr kernelCtr)
{
    enforce((opName in mKernelCtrs) is null,
        "A CUDAKernelCtr is already registered for the operation '" ~ opName ~ "'");

    mKernelCtrs[opName] = kernelCtr;
}

/**
    Deregisters a kernel constructor associated with the given operation type.

    Params:
        opType = The operation type that should have its kernel deregistered.
*/
void deregisterCUDAKernel(string opType)
{
    mKernelCtrs.remove(opType);
}

/**
    Provides a list of all operation types supported by the CUDA backend.

    Returns:
        A string array of the operation types that have kernels registered.
*/
string[] listCUDAOperations()
{
    return mKernelCtrs.keys ~ ["variable", "reshape"];
}

package
{
    string cudaType(DataType t)
    {
        switch(t)
        {
            case DataType.float32:
                return "float";
            
            case DataType.int32:
                return "int";

            default:
                import std.conv : to;
                assert(0, "DataType '" ~ t.to!string ~ "' is not currently supported by the CUDA backend");
        }
    }
}

private
{
    CUDAKernelCtr[string] mKernelCtrs;
}

unittest
{
    auto a = float32([], [3.0f]);
    auto b = float32([], [4.0f]);
    auto c = float32([], [-1.0f]);

    auto y = a * b + c;

    assert(evaluateCUDA(y).get!float[0] == 11.0f);
}