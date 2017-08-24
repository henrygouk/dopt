/**
    This is the main interface for the dopt CUDA backend.

    The APIs in this module allow users to evaluate operation graphs on GPUs through the use of CUDA. There is also
    functionality to register CUDA implementations of custom operations.

    In future, this module will also have an interface allowing the user to register their own optimisation passes to
    be called when constructing a plan.

    Authors: Henry Gouk
*/
module dopt.core.cuda;

import std.exception;

import dopt.core.cuda.basic;
import dopt.core.cuda.nvrtc;
import dopt.core.cuda.math;
import dopt.core.cuda.nnet;
import dopt.core.ops;
import dopt.core.types;

import derelict.cuda;

alias CUDAKernelCtr = CUDAKernel delegate(Operation op);

private __gshared
{
    CUdevice mDevice;
    CUcontext mContext;
}

void initialize()
{
    //TODO: handle case where CUDA isn't available
    DerelictCUDADriver.load();
    
    //Initialise CUDA and create a context
    cuInit(0);
    cuDeviceGet(&mDevice, 0);
    cuCtxCreate(&mContext, 0, mDevice);

    //Initialize submodules
    dopt.core.cuda.basic.initialize();
    dopt.core.cuda.nvrtc.initialize();
    dopt.core.cuda.math.initialize();
    dopt.core.cuda.nnet.initialize();
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

/**
    A class that encapsulates the CUDA memory allocation/deallocation process.
*/
class CUDABuffer
{
    public
    {
        /**
            Constructs a CUDABuffer object and allocates memory on the CUDA device.

            Params:
                numBytes = The number of bytes to be allocated on the CUDA device.
        */
        this(size_t numBytes)
        {
            mNumBytes = numBytes;
            enforce(cuMemAlloc(&mPtr, mNumBytes) == CUDA_SUCCESS, "CUDA memory allocation failed");
            enforce(cuMemsetD8(mPtr, 0, mNumBytes) == CUDA_SUCCESS, "CUDA default buffer initialisation failed");
        }

        ~this()
        {
            cuMemFree(mPtr);
        }

        /**
            Copies data from the host to the device.

            Params:
                buf = An array of data to be copied to the device.
        */
        void set(void[] buf)
        {
            enforce(buf.length == mNumBytes, "input buffer is the wrong length.");
			enforce(cuMemcpyHtoD(mPtr, buf.ptr, buf.length) == CUDA_SUCCESS, "Failed to set contents of CUDA buffer");
        }

        /**
            Copies data from the device to the host.

            Params:
                buf = The buffer that the data from the CUDA device will be written to.
        */
        void get(void[] buf) const
        {
            enforce(buf.length == mNumBytes, "output buffer is the wrong length.");
			enforce(cuMemcpyDtoH(buf.ptr, mPtr, buf.length) == CUDA_SUCCESS, "Failed to get contents of CUDA buffer");
        }

        /**
            Provides the size of the buffer allocated on the CUDA device.

            Returns:
                The number of bytes allocated on the CUDA device.
        */
        size_t numBytes() const
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
    }
}

/**
    A Plan stores all the resources (preallocated buffers, custom CUDA kernels) required to evaluate nodes from the
    Operation graph.

    An instance of Plan can be constructed using the $(D compileCUDA) function. The primary use case for a CUDAPlan is when the
    same set of operations are likely to be evaluated more than once. This prevents the dopt CUDA runtime from
    reallocating and optimising the CUDA kernels every time the same set of operations is to be executed.
*/
class CUDAPlan
{
    public
    {
        /**
            Executes the plan.

            Params:
                args = A set of variable assignments.
        */
        Buffer[] execute(Buffer[Operation] args = null)
        {
            //Make sure all the args are variable assignments
            foreach(o; args.keys)
            {
                enforce(o.opType == "variable",
                    "All assignments in args must be for Operations with an opType of 'variable'");
            }

            //Convert the args into CUDABuffer objects
            CUDABuffer[Operation] results;

            foreach(k, v; args)
            {
                auto buf = new CUDABuffer(v.as!ubyte.length);
                buf.set(v.as!ubyte);

                results[k] = buf;
            }

            //Count the number of references to each operation
            int[Operation] refCounts;

            foreach(o; mOps)
            {
                foreach(d; o.deps)
                {
                    refCounts[d]++;
                }
            }

            //Iterate through each operation and execute it
            foreach(o; mOps)
            {
                //Check for some easy optimizations
                if(o.opType == "variable" && !(o in mKernels))
                {
                    auto buf = cast(Buffer)o.value;
                    auto cbuf = new CUDABuffer(buf.as!ubyte.length);
                    cbuf.set(buf.as!ubyte);
                    results[o] = cbuf;
                    continue;
                }
                else if(o.opType == "reshape" && !(o in mKernels))
                {
                    results[o] = results[o.deps[0]];
                    continue;
                }

                //Allocate a buffer for the output of this operation
                auto output = new CUDABuffer(o.outputType.volume * o.outputType.elementType.sizeOf());
                results[o] = output;

                //Get the input buffers
                CUDABuffer[] inputs;

                foreach(d; o.deps)
                {
                    inputs ~= results[d];
                    refCounts[d]--;
                }

                //Execute the operation
                mKernels[o].execute(inputs, output);

                foreach(d; o.deps)
                {
                    //Remove the pointer to this buffer if we don't need it anymore
                    //This will allow the GC to collect it at some point, if required
                    if(refCounts[d] == 0)
                    {
                        results.remove(d);
                    }
                }
            }

            //Convert the results into regular Buffer objects
            Buffer[] rets = new Buffer[mOutputs.length];

            foreach(i, o; mOutputs)
            {
                rets[i] = Buffer(new ubyte[o.outputType.volume * o.outputType.elementType.sizeOf()]);
                results[o].get(rets[i].as!ubyte);
            }

            return rets;
        }
    }

    private
    {
        Operation[] mOutputs;
        Operation[] mOps;
        CUDAKernel[Operation] mKernels;

        this(Operation[] ops, Operation[] outputs, CUDAKernel[Operation] kernels)
        {
            import std.array : array;

            mOps = ops.array;
            mOutputs = outputs.array;
            mKernels = kernels;
        }
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
Buffer[] evaluateCUDA(Operation[] ops, Buffer[Operation] args = null)
{
    import std.algorithm : canFind, filter;
    import std.array : array;

    auto sortedOps = topologicalSort(ops)
                    .filter!(x => !canFind(args.keys, x))
                    .array();

    CUDAKernel[Operation] kernels;

    foreach(o; sortedOps)
    {
        if(o.opType == "variable" || o.opType == "reshape")
        {
            continue;
        }
        
        auto k = mKernelCtrs.get(o.opType, null);

        enforce(k !is null, "Could not construct a CUDA kernel for operation of type '" ~ o.opType ~ "'");

        kernels[o] = k(o);
    }

    auto p = new CUDAPlan(sortedOps, ops, kernels);
    
    return p.execute(args);
}

/**
    A convenience overload that evaluates a single operation and returns a single $(D Buffer).

    Params:
        op = The operation to be evaluated.
        args = A set of optional variable assignments.

    Returns:
        The result of evaluating $(D op)
*/
Buffer evaluateCUDA(Operation op, Buffer[Operation] args = null)
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
                return "int32";

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

    assert(evaluateCUDA(y).as!float[0] == 11.0f);
}