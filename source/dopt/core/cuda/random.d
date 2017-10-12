module dopt.core.cuda.random;

import std.algorithm;
import std.conv;
import std.functional;
import std.math;
import std.random;
import std.range;

import dopt.core.cuda;
import dopt.core.cuda.nvrtc;
import dopt.core.ops;
import dopt.core.types;

package
{
    extern(C)
    {
        struct curandGenerator_st;
        alias curandGenerator_t = curandGenerator_st *;

        immutable int CURAND_RNG_PSEUDO_DEFAULT = 100;

        int function(curandGenerator_t *generator, int rng_type) curandCreateGenerator;
        int function(curandGenerator_t generator) curandDestroyGenerator;
        int function(curandGenerator_t generator, ulong seed) curandSetPseudoRandomGeneratorSeed;
        int function(curandGenerator_t generator, float *outputPtr, size_t num) curandGenerateUniform;
    }

    void initialize()
    {
        //TODO: make a DerelictCuRAND library
        import core.sys.posix.dlfcn;
		import std.string;
		fh = dlopen("libcurand.so".toStringz, RTLD_LAZY);

        curandCreateGenerator = cast(typeof(curandCreateGenerator))dlsym(fh, "curandCreateGenerator");
        curandDestroyGenerator = cast(typeof(curandDestroyGenerator))dlsym(fh, "curandDestroyGenerator");
        curandSetPseudoRandomGeneratorSeed =
            cast(typeof(curandSetPseudoRandomGeneratorSeed))dlsym(fh, "curandSetPseudoRandomGeneratorSeed");
        curandGenerateUniform = cast(typeof(curandGenerateUniform))dlsym(fh, "curandGenerateUniform");

        registerCUDAKernel("uniform", toDelegate(&uniformCtor));
    }
}

private
{
    void *fh;

    CUDAKernel uniformCtor(Operation op)
    {
        return new UniformSample(op);
    }

    class UniformSample : CUDAKernel
    {
        public
        {
            this(Operation op)
            {
                mOp = op;
                curandCreateGenerator(&mGen, CURAND_RNG_PSEUDO_DEFAULT);
            }

            ~this()
            {
                curandDestroyGenerator(mGen);
            }

            override void execute(const(CUDABuffer)[] inputs, CUDABuffer output)
            {
                curandSetPseudoRandomGeneratorSeed(mGen, cast(ulong)unpredictableSeed());
                curandGenerateUniform(mGen, cast(float *)output.ptr, mOp.volume);
            }
        }

        private
        {
            Operation mOp;
            curandGenerator_t mGen;
        }
    }
}