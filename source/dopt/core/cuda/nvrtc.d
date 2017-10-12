module dopt.core.cuda.nvrtc;

import std.algorithm;
import std.array;
import std.string;

import derelict.cuda;
import derelict.nvrtc;

package
{
    void initialize()
    {
        DerelictNVRTC.load();
    }
}

private
{
    void nvrtcCheck(nvrtcResult res, string mod = __MODULE__, size_t line = __LINE__)
    {
        import std.conv : to;
        import std.exception : enforce;
        enforce(res == NVRTC_SUCCESS, mod ~ "(" ~ line.to!string ~ "): Failed to execute NVRTC function.");
    }
}

class NVRTCKernel
{
    public
    {
        /**
            Constructs an NVRTCKernel with the given code and entry point.

            Params:
                entry = The name of the function inside the CUDA code that should b executed.
                code = A string containing the CUDA code to be compiled.
        */
        this(string entry, string code)
        {
            immutable(char) *entryz = entry.toStringz;
            immutable(char) *codez = code.toStringz;

            nvrtcProgram program;

			auto options = [//"compute_20",
							//"compute_30",
							"compute_35"//,
							//"compute_50",
							//"compute_52",
							/*"compute_53"*/].map!(x => ("--gpu-architecture=" ~ x).toStringz()).array();

			nvrtcCreateProgram(&program, codez, entryz, 0, null, null).nvrtcCheck;
			nvrtcCompileProgram(program, cast(int)options.length, options.ptr).nvrtcCheck;

			size_t logSize;
			nvrtcGetProgramLogSize(program, &logSize).nvrtcCheck;

			if(logSize > 1)
			{
				auto log = new char[logSize];
				nvrtcGetProgramLog(program, log.ptr).nvrtcCheck;

				import std.stdio;
				stderr.writeln(log[0 .. $ - 1]);
			}

			size_t ptxSize;
			nvrtcGetPTXSize(program, &ptxSize).nvrtcCheck;

			auto ptx = new char[ptxSize];
			nvrtcGetPTX(program, ptx.ptr).nvrtcCheck;
			nvrtcDestroyProgram(&program).nvrtcCheck;

			cuModuleLoadDataEx(&mModule, ptx.ptr, 0, null, null);
			cuModuleGetFunction(&mKernel, mModule, entryz);
        }

        void execute(Args...)(uint numBlocks, uint numThreads, Args args)
        {
            execute([numBlocks, 1, 1], [numThreads, 1, 1], args);
        }

        void execute(Args...)(uint[3] numBlocks, uint[3] numThreads, Args args)
        {
            void*[] argPtrs = new void*[args.length];

            string genCode()
            {
                import std.conv : to;
                string ret;

                for(size_t i = 0; i < args.length; i++)
                {
                    ret ~= "argPtrs[" ~ i.to!string ~ "] = cast(void *)&args[" ~ i.to!string ~ "];";
                }

                return ret;
            }

            mixin(genCode());
            
            cuLaunchKernel(
                mKernel,
                numBlocks[0], numBlocks[1], numBlocks[2],
                numThreads[0], numThreads[1], numThreads[2],
                0, null, argPtrs.ptr, null
            );

            cuCtxSynchronize();
        }

        void execute(uint[3] numBlocks, uint[3] numThreads, void*[] argPtrs)
        {
            cuLaunchKernel(
                mKernel,
                numBlocks[0], numBlocks[1], numBlocks[2],
                numThreads[0], numThreads[1], numThreads[2],
                0, null, argPtrs.ptr, null
            );

            cuCtxSynchronize();
        }
    }

    private
    {
        CUmodule mModule;
        CUfunction mKernel;
    }
}
