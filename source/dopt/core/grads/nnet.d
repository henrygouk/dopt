module dopt.core.grads.nnet;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional : toDelegate;
        
        registerGradient("convolution", toDelegate(&convolutionGrad));
        registerGradient("convolutionFeaturesGrad", toDelegate(&convolutionFeaturesGradGrad));
        registerGradient("maxpool", toDelegate(&maxpoolGrad));
        registerGradient("softmax", toDelegate(&softmaxGrad));
    }
}

private
{
    Operation[] convolutionGrad(Operation op, Operation parentGrad)
    {
        return [
            convolutionFeaturesGrad(parentGrad, op.deps[1], op.deps[0].shape),
            convolutionFiltersGrad(parentGrad, op.deps[0], op.deps[1].shape)
        ];
    }

    Operation[] convolutionFeaturesGradGrad(Operation op, Operation parentGrad)
    {
        return [
            convolution(parentGrad, op.deps[1]),
            convolutionFiltersGrad(parentGrad, op.deps[0], op.deps[1].shape)
        ];
    }

    Operation[] maxpoolGrad(Operation op, Operation parentGrad)
    {
        return [dopt.core.ops.nnet.maxpoolGrad(parentGrad, op)];
    }

    Operation[] softmaxGrad(Operation op, Operation parentGrad)
    {
        return [dopt.core.ops.nnet.softmaxGrad(parentGrad, op)];
    }
}