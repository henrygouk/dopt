/**
    This module contains methods for initialising the parameters of neural networks.

    Several of the methods implemented in this module rely on $(D fan_in) and $(D fan_out) values. These are calculated
    differently depending on the rank of the parameter.

    For rank-2 tensors,
        
        $(D fan_in = shape[0]), $(D fan_out = shape[1])
    
    For rank-4 tensors,
        
        $(D fan_in = shape[1] * shape[2] * shape[3]), $(D fan_out = shape[0] * shape[2] * shape[3])
*/
module dopt.nnet.parameters;

import std.math;

import dopt.core;
import dopt.online;

/**
    Used to initialize a parameter in the neural network.

    The $(D param) parameter will contain an $(D Operation) representing a variable. The ParamInitializer will set the
    default value of this variable according to some parameter initialisation scheme.
*/
alias ParamInitializer = void delegate(Operation param);

private
{
    void fillUniform(float[] vals, float minval, float maxval)
    {
        import std.random : uniform;

        for(size_t i = 0; i < vals.length; i++)
        {
            vals[i] = uniform(minval, maxval);
        }
    }

    void fillGaussian(float[] vals, float mean, float stddev)
    {
        import std.mathspecial : normalDistributionInverse;
        import std.random : uniform;

        for(size_t i = 0; i < vals.length; i++)
        {
            vals[i] = normalDistributionInverse(uniform(0.0f, 1.0f)) * stddev + mean;
        }
    }

    size_t fanIn(size_t[] shape)
    {
        if(shape.length == 2)
        {
            return shape[1];
        }
        else if(shape.length == 4)
        {
            return shape[1] * shape[2] * shape[3];
        }
        else
        {
            import std.conv : to;
            throw new Exception("Cannot compute fan-in for a parameter tensor of rank " ~ shape.length.to!string);
        }
    }

    size_t fanOut(size_t[] shape)
    {
        if(shape.length == 2)
        {
            return shape[0];
        }
        else if(shape.length == 4)
        {
            return shape[0] * shape[2] * shape[3];
        }
        else
        {
            import std.conv : to;
            throw new Exception("Cannot compute fan-out for a parameter tensor of rank " ~ shape.length.to!string);
        }
    }
}

/**
    Encapsulates information about network parameters.

    This can be used to keep track of per-parameter loss functions (e.g., weight decay), and also projection functions
    that can be applied using constrained optimisation methods.
*/
struct Parameter
{
    ///An "variable" operation.
    Operation symbol;

    ///Used for applying loss terms to this parameter (e.g., weight decay)
    Operation loss;

    ///A projection operation that can enforce some constraint
    Projection projection;
}

/**
    Creates a parameter initialiser that sets the initial value of each element in a parameter tensor to a constant
    value.

    Params:
        val = The constant value to be used for initialisation.

    Returns:
        The constructed $(D ParamInitializer).
*/
ParamInitializer constantInit(float val)
{
    void init(Operation param)
    {
        import std.array : array;
        import std.range : repeat;

        param.value.set(repeat(val, param.volume).array());
    }

    return &init;
}

/**
    Creates a parameter initialiser that sets the initial value of each element in a parameter tensor to a different
    sample from a uniform distribution.

    Params:
        minval = The lower bound of the uniform distribution.
        maxval = The upper bound of the uniform distribution.
    
    Returns:
        The constructed $(D ParamInitializer).
*/
ParamInitializer uniformInit(float minval, float maxval)
{
    void init(Operation param)
    {
        auto buf = param.value.get!float;
        fillUniform(buf, minval, maxval);
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a parameter initialiser that sets the initial value of each element in a parameter tensor to a different
    sample from a Gaussian distribution.

    Params:
        mean = The mean of the Gaussian distribution.
        stddev = The standard deviation of the Gaussian distribution.
    
    Returns:
        The constructed $(D ParamInitializer).
*/
ParamInitializer gaussianInit(float mean, float stddev)
{
    void init(Operation param)
    {
        auto buf = param.value.get!float;
        fillGaussian(buf, mean, stddev);
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a parameter initialiser that uses the method of Glorot and Bengio (2010).

    This technique initialises a parameter with samples from the following uniform distribution:

    U(-6 / (fan_in + fan_out), 6 / (fan_in + fan_out))

    For more details, see Glorot and Bengio (2010): http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    Returns:
        The constructed $(D ParamInitiaizer).
*/
ParamInitializer glorotUniformInit()
{
    void init(Operation param)
    {
        auto bound = sqrt(6.0f / (param.shape.fanIn + param.shape.fanOut));
        auto buf = param.value.get!float;
        fillUniform(buf, -bound, bound);
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a parameter initialiser that uses the method of Glorot and Bengio (2010).

    This technique initialises a parameter with samples from the following Gaussian distribution:

    μ = 0
    σ = sqrt(2 / (fan_in + fan_out))

    For more details, see Glorot and Bengio (2010): http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf

    Returns:
        The constructed $(D ParamInitiaizer).
*/
ParamInitializer glorotGaussianInit()
{
    void init(Operation param)
    {
        auto buf = param.value.get!float;
        fillGaussian(buf, 0, sqrt(2.0f / (param.shape.fanIn + param.shape.fanOut)));
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a parameter initialiser that uses the method of He et al. (2015).

    This technique initialises a parameter with samples from the following uniform distribution:

    U(-6 / fan_in, 6 / fan_in)

    For more details, see He et al. (2015): http://arxiv.org/abs/1502.01852

    Returns:
        The constructed $(D ParamInitiaizer).
*/  
ParamInitializer heUniformInit()
{
    void init(Operation param)
    {
        auto buf = param.value.get!float;
        fillUniform(buf, 0, sqrt(6.0f / (param.shape.fanIn)));
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a parameter initialiser that uses the method of He et al. (2015).

    This technique initialises a parameter with samples from the following Gaussian distribution:

    μ = 0
    σ = sqrt(2 / fan_in)

    For more details, see He et al. (2015): http://arxiv.org/abs/1502.01852

    Returns:
        The constructed $(D ParamInitiaizer).
*/  
ParamInitializer heGaussianInit()
{
    void init(Operation param)
    {
        auto buf = param.value.get!float;
        fillUniform(buf, 0, sqrt(2.0f / (param.shape.fanIn)));
        param.value.set(buf);
    }

    return &init;
}

/**
    Creates a $(D Projection) function that can be applied to a parameter matrix/tensor to constraint the Lipschitz
    constant w.r.t. the L_1 vector norm.

    This function should be used to create a $(D Projection) for $(D Conv2DOptions.filterProj) or
    $(D DenseOptions.weightProj).

    See https://arxiv.org/abs/1804.04368 for more details.

    Params:
        maxK = The maximum permissible Lipschitz constant.
    
    Returns:
        A $(D Projection) function that can be used to enforce the constraint.
*/
Projection lipschitz1(float maxK)
{
    Operation proj(Operation param)
    {
        import dopt.core : abs;

        if(param.rank != 2)
        {
            param = param.reshape([param.shape[0], param.volume / param.shape[0]]);
        }

        /*
            The matrix norm induced by the L1 vector norm is max Sum_i abs(a_ij), which is the maximum absolute column
            sum. We are doing a row sum here because the weight matrices in dense and convolutional layers are
            transposed before being multiplied with the input features.
        */
        auto norm = param.abs().sum([1]).maxElement();

        return param * (1.0f / max(float32([], [1.0f]), norm / maxK));
    }

    return &proj;
}

/**
    Creates a $(D Projection) function that can be applied to a parameter matrix/tensor to constraint the Lipschitz
    constant w.r.t. the L_infty vector norm.

    This function should be used to create a $(D Projection) for $(D Conv2DOptions.filterProj) or
    $(D DenseOptions.weightProj).

    See https://arxiv.org/abs/1804.04368 for more details.

    Params:
        maxK = The maximum permissible Lipschitz constant.
    
    Returns:
        A $(D Projection) function that can be used to enforce the constraint.
*/
Projection lipschitzInf(float maxK)
{
    Operation proj(Operation param)
    {
        import dopt.core : abs;
        
        if(param.rank != 2)
        {
            param = param.reshape([param.shape[0], param.volume / param.shape[0]]);
        }

        /*
            The matrix norm induced by the L_infty vector norm is max Sum_j abs(a_ij), which is the maximum absolute
            row sum. We are doing a column sum here because the weight matrices in dense and convolutional layers are
            transposed before being multiplied with the input features.
        */
        auto norm = param.abs().sum([0]).maxElement();

        return param * (1.0f / max(float32([], [1.0f]), norm / maxK));
    }

    return &proj;
}