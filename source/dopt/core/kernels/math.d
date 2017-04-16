module dopt.core.kernels.math;

import dopt.core;

static this()
{
    registerKernel("add", new Kernel(Device.cpu));
}