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
        registerGradient("addBias", toDelegate(&addBiasGrad));
        registerGradient("batchNormTrain", toDelegate(&batchNormGrad));
    }
}

private
{
    Operation[] convolutionGrad(Operation op, Operation parentGrad)
    {
        auto padding = op.attributes["padding"].get!(size_t[]);
        auto stride = op.attributes["stride"].get!(size_t[]);

        return [
            convolutionFeaturesGrad(parentGrad, op.deps[1], op.deps[0].shape, padding, stride),
            convolutionFiltersGrad(parentGrad, op.deps[0], op.deps[1].shape, padding, stride)
        ];
    }

    Operation[] convolutionFeaturesGradGrad(Operation op, Operation parentGrad)
    {
        auto padding = op.attributes["padding"].get!(size_t[]);
        auto stride = op.attributes["stride"].get!(size_t[]);

        return [
            convolution(parentGrad, op.deps[1], padding, stride),
            convolutionFiltersGrad(parentGrad, op.deps[0], op.deps[1].shape, padding, stride)
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

    Operation[] addBiasGrad(Operation op, Operation parentGrad)
    {
        return [parentGrad, dopt.core.ops.nnet.addBiasGrad(parentGrad)];
    }

    Operation[] batchNormGrad(Operation op, Operation parentGrad)
    {
        auto packedGrads = dopt.core.ops.nnet.batchNormGrad(parentGrad, op.deps[0], op.deps[1]);
        packedGrads = packedGrads.reshape([packedGrads.volume]);

        auto v0 = op.deps[0].volume;
        auto v1 = op.deps[1].volume;
        auto v2 = op.deps[2].volume;

        return [
            packedGrads.slice([0], [v0]).reshape(op.deps[0].shape),
            packedGrads.slice([v0], [v0 + v1]).reshape(op.deps[1].shape),
            packedGrads.slice([v0 + v1], [v0 + v1 + v2]).reshape(op.deps[2].shape)
        ];
    }
}