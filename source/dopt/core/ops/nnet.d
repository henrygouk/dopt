/**
    Contains constructors for common neural network operations.

    These operations are currently only implemented for the CUDA backend.

    Authors: Henry Gouk
*/
module dopt.core.ops.nnet;

import dopt.core.ops;
import dopt.core.types;

import std.array;
import std.functional;
import std.variant;

package
{
    void initialize()
    {
        registerOperation("convolution", OpDef(toDelegate(&verifyConvolution), toDelegate(&judgeConvolution)));
        registerOperation("maxpool", OpDef(toDelegate(&verifyMaxpool), toDelegate(&judgeMaxpool)));
        registerOperation("convolutionFeaturesGrad", OpDef(toDelegate(&verifyConvolutionFeaturesGrad),
            toDelegate(&judgeConvolutionFeaturesGrad)));
        registerOperation("convolutionFiltersGrad", OpDef(toDelegate(&verifyConvolutionFiltersGrad),
            toDelegate(&judgeConvolutionFiltersGrad)));
        registerOperation("maxpoolGrad", OpDef(toDelegate(&verifyMaxpoolGrad), toDelegate(&judgeMaxpoolGrad)));
        registerOperation("softmax", OpDef(toDelegate(&verifySoftmax), toDelegate(&judgeSoftmax)));
        registerOperation("softmaxGrad", OpDef(toDelegate(&verifySoftmaxGrad), toDelegate(&judgeSoftmaxGrad)));
    }
}

private
{
    bool verifyConvolution(Operation op)
    {
        if(op.deps.length != 2)
        {
            return false;
        }

        auto imgs = op.deps[0].outputType;
        auto filters = op.deps[1].outputType;

        if(imgs.rank != 4 || filters.rank != 4)
        {
            return false;
        }

        if(imgs.elementType != filters.elementType)
        {
            return false;
        }

        if(imgs.shape[1] != filters.shape[1])
        {
            return false;
        }

        if(imgs.shape[2] < filters.shape[2] || imgs.shape[3] < filters.shape[3])
        {
            return false;
        }

        return true;
    }

    TensorType judgeConvolution(Operation op)
    {
        auto imgs = op.deps[0];
        auto filters = op.deps[1];

        auto batchSize = imgs.outputType.shape[0];
        auto outputChannels = filters.outputType.shape[0];
        auto newHeight = imgs.outputType.shape[2] - filters.outputType.shape[2] + 1;
        auto newWidth = imgs.outputType.shape[3] - filters.outputType.shape[3] + 1;

        auto shape = [batchSize, outputChannels, newHeight, newWidth];

        return TensorType(imgs.outputType.elementType, shape);
    }

    bool verifyMaxpool(Operation op)
    {
        return op.deps.length == 1
            && op.deps[0].outputType.rank == 4
            && op.attributes["dims"].peek!(size_t[]) !is null
            && op.attributes["dims"].get!(size_t[]).length == 2;
    }

    TensorType judgeMaxpool(Operation op)
    {
        auto poolDims = op.attributes["dims"].get!(size_t[]);
        size_t[] shape = new size_t[4];
        shape[0] = op.deps[0].shape[0];
        shape[1] = op.deps[0].shape[1];
        shape[2] = op.deps[0].shape[2] / poolDims[0];
        shape[3] = op.deps[0].shape[3] / poolDims[1];

        return TensorType(op.deps[0].outputType.elementType, shape);
    }

    bool verifyConvolutionFeaturesGrad(Operation op)
    {
        return true;
    }

    TensorType judgeConvolutionFeaturesGrad(Operation op)
    {
        auto parentGrad = op.deps[0];
        auto dims = op.attributes["featuresShape"].get!(size_t[]);

        size_t[] shape = new size_t[4];
        shape[] = dims[];

        return TensorType(parentGrad.outputType.elementType, shape);
    }

    bool verifyConvolutionFiltersGrad(Operation op)
    {
        return true;
    }

    TensorType judgeConvolutionFiltersGrad(Operation op)
    {
        auto parentGrad = op.deps[0];
        auto dims = op.attributes["filtersShape"].get!(size_t[]);

        size_t[] shape = new size_t[4];
        shape[] = dims[];

        return TensorType(parentGrad.outputType.elementType, shape);
    }

    bool verifyMaxpoolGrad(Operation op)
    {
        return true;
    }

    TensorType judgeMaxpoolGrad(Operation op)
    {
        auto parentGrad = op.deps[0];
        auto dims = op.attributes["featuresShape"].get!(size_t[]);

        size_t[] shape = new size_t[4];
        shape[] = dims[];

        return TensorType(parentGrad.outputType.elementType, shape);
    }

    bool verifySoftmax(Operation op)
    {
        return op.deps.length == 1;
    }

    TensorType judgeSoftmax(Operation op)
    {
        return TensorType(op.deps[0].elementType, op.deps[0].shape);
    }

    bool verifySoftmaxGrad(Operation op)
    {
        return op.deps.length == 2;
    }

    TensorType judgeSoftmaxGrad(Operation op)
    {
        return TensorType(op.deps[1].elementType, op.deps[1].shape);
    }
}

public
{
    /**
        Creates a convolution operation that performs the computation required to implement a convolutional layer.

        Currently this operation only implements 2D convolutions.

        Params:
            features = A tensor containing a batch of input feature maps.
            filters = A tensor containing the filters that will be convolved with the feature maps.
        
        Returns:
            An operation representing convolutions of input imgs with some kernels.
    */
    Operation convolution(Operation features, Operation filters, string mod = __MODULE__,
        size_t line = __LINE__)
    {
        return createOperation("convolution", [features, filters], null, mod, line);
    }

    ///
    unittest
    {
        import dopt.core.cuda : evaluateCUDA;

        auto features = float32([1, 1, 3, 5], [
            1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 0.0f, 0.0f,
            1.0f, 1.0f, 1.0f, 0.0f, 0.0f
        ]);

        auto filters = float32([1, 1, 1, 3], [
            -1.0f, 1.0f
        ]);

        auto result = convolution(features, filters);

        auto edges = result.evaluateCUDA().as!float;

        assert(edges == [
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f,
            0.0f, 1.0f, 0.0f
        ]);
    }

    /**
        Creates a transposed convolution operation (also known, incorrectly, as deconvolution).

        Params:
            features = The feature maps.
            filters = The filters to be convolved with the feature maps.
        
        Returns:
            The operation.
    */
    Operation convolutionTranspose(Operation features, Operation filters, string mod = __MODULE__,
        size_t line = __LINE__)
    {
        auto outShape = features.shape.dup;
        outShape[] += filters.shape[] - 1;

        return convolutionFeaturesGrad(features, filters, outShape, mod, line);
    }

    /**
        Creates a max pool operation that performs the computation required to implement a max pooling layer.

        Params:
            features = A tensor containing a batch of input feature maps.
            dims = An array of pool dims.

        Returns:
            An operation representing a max pool computation.
    */
    Operation maxpool(Operation features, size_t[] dims, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("maxpool", [features], ["dims": Variant(dims)], mod, line);
    }

    ///
    unittest
    {
        import dopt.core.cuda : evaluateCUDA;

        auto features = float32([1, 1, 4, 4], [
            1.0f, 2.0f, 4.0f, 3.0f,
            5.0f, 3.0f, 2.0f, 2.0f,
            0.1f, -4.0f, 3.0f, 2.0f,
            0.0f, 0.0f, 2.0f, 2.0f
        ]);

        auto result = features.maxpool([2,2]);

        auto pooledFeatures = result.evaluateCUDA().as!float;

        assert(pooledFeatures == [
            5.0f, 4.0f,
            0.1f, 3.0f
        ]);
    }

    /**
        Creates an operation representing the derivative of a convolution operation with respect to the feature maps.

        Params:
            parentGrad = Gradient of some functions w.r.t. the convolution operation.
            filters = The filters of the convolution operation.
            featuresShape = The shape of the features fed into the convolution operations.
        
        Returns:
            The gradient.
    */
    Operation convolutionFeaturesGrad(Operation parentGrad, Operation filters, size_t[] featuresShape,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("convolutionFeaturesGrad", [parentGrad, filters],
            ["featuresShape": Variant(featuresShape)], mod, line);
    }

    /**
        Creates an operation representing the derivative of a convolution operation with respect to the filters.

        Params:
            parentGrad = Gradient of some functions w.r.t. the convolution operation.
            features = The features provided to the convolution operation.
            filtersShape = The shape of the filters provided to the convolution operation.
        
        Returns:
            The gradient.
    */
    Operation convolutionFiltersGrad(Operation parentGrad, Operation features, size_t[] filtersShape,
        string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("convolutionFiltersGrad", [parentGrad, features],
            ["filtersShape": Variant(filtersShape)], mod, line);
    }

    /**
        Creates an operation representing the derivative of a maxpool operation with respect to the feature maps.

        Params:
            parentGrad = Gradient of some function w.r.t. the maxpool operation.
            op = The operation being differentiated.

        Returns:
            The gradient.
    */
    Operation maxpoolGrad(Operation parentGrad, Operation op, string mod = __MODULE__,
        size_t line = __LINE__)
    {
        return createOperation("maxpoolGrad", [parentGrad, op, op.deps[0]],
            ["featuresShape": Variant(op.deps[0].outputType.shape), "dims": op.attributes["dims"]], mod, line);
    }

    /**
        Creates an operation representing the computation required for a softmax layer.

        Params:
            inputs = The inputs to the softmax function.
        
        Returns:
            The operation.
    */
    Operation softmax(Operation inputs, string mod = __MODULE__, size_t line = __LINE__)
    {
        return createOperation("softmax", [inputs], null, mod, line);
    }

    ///
    unittest
    {
        import std.math : approxEqual;
        import dopt.core.cpu : evaluateCUDA;

        auto y = float32([1, 5], [1.0f, 2.0f, 3.0f, 1.0f, 2.0f]).softmax();

        assert(approxEqual(
            y.evaluateCUDA().as!float,
            [0.0674508, 0.18335, 0.498398, 0.0674508, 0.18335]
        ));
    }

    /**
        Creates an operation representing the gradient of the softmax function.
    */
    Operation softmaxGrad(Operation parentGrad, Operation op, string mod = __MODULE__,
        size_t line = __LINE__)
    {
        return createOperation("softmaxGrad", [parentGrad, op], null, mod, line);
    }
}
