module dopt.core.grads.nnet;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional : toDelegate;
        
        registerGradient("convolution", toDelegate(&convolutionGrad));
        registerGradient("maxpool", toDelegate(&maxpoolGrad));
        registerGradient("softmax", toDelegate(&softmaxGrad));
    }
}

private
{
    Operation[] convolutionGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            convolutionFeaturesGrad(parentGrad, op),
            convolutionFiltersGrad(parentGrad, op)
        ];
    }

    Operation[] maxpoolGrad(const(Operation) op, Operation parentGrad)
    {
        return [dopt.core.ops.nnet.maxpoolGrad(parentGrad, op)];
    }

    Operation[] softmaxGrad(const(Operation) op, Operation parentGrad)
    {
        return [dopt.core.ops.nnet.softmaxGrad(parentGrad, op)];
    }
}