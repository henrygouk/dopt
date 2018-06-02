/**
    Contains an implementation of the regularisation techniques presented in Gouk et al. (2018).

    Gouk, H., Frank, E., Pfahringer, B., & Cree, M. (2018). Regularisation of Neural Networks by Enforcing Lipschitz
    Continuity. arXiv preprint arXiv:1804.04368.

    Authors: Henry Gouk
*/
module dopt.nnet.lipschitz;

import std.exception;

import dopt.core;
import dopt.online;

/**
    Returns a projection function that can be used to constrain a matrix norm.

    The operator norm induced by the vector p-norm is used.

    Params:
        maxnorm = A scalar value indicating the maximum allowable operator norm.
        p = The vector p-norm that will induce the operator norm.
    
    Returns:
        A projection function that can be used with the online optimisation algorithms.
*/
Projection projMatrix(Operation maxnorm, float p = 2)
{
    Operation proj(Operation param)
    {
        auto norm = matrixNorm(param, p);

        return maxNorm(param, norm, maxnorm);
    }

    return &proj;
}

/**
    Computes the induced operator norm corresponding to the vector p-norm.
*/
Operation matrixNorm(Operation param, float p, size_t n = 2)
{
    import std.exception : enforce;

    enforce(param.rank == 2, "This function only operates on matrices");

    if(p == 1.0f)
    {
        /*if(param.rank != 2)
        {
            param = param.reshape([param.shape[0], param.volume / param.shape[0]]);
        }*/

        /*
            The matrix norm induced by the L1 vector norm is max Sum_i abs(a_ij), which is the maximum absolute column
            sum. We are doing a row sum here because the weight matrices in dense and convolutional layers are
            transposed before being multiplied with the input features.
        */

        return param.abs().sum([1]).maxElement();
    }
    else if(p == 2.0f)
    {
        auto x = uniformSample([param.shape[0], 1]) * 2.0f - 1.0f;

        for(int i = 0; i < n; i++)
        {
            x = matmul(param, matmul(param.transpose([1, 0], x));
        }

        auto v = x / sqrt(sum(x * x));
        auto y = matmul(param, v);

        return sqrt(sum(y * y));
    }
    else if(p == float.infinity)
    {
        /*
            The matrix norm induced by the L_infty vector norm is max Sum_j abs(a_ij), which is the maximum absolute
            row sum. We are doing a column sum here because the weight matrices in dense and convolutional layers are
            transposed before being multiplied with the input features.
        */

        return param.abs().sum([0]).maxElement();
    }
    else
    {
        import std.conv : to;

        throw new Exception("Cannot compute matrix norm for p=" ~ p.to!string);
    }
}

Projection projConvParams(Operation maxnorm, size_t[] inShape, size_t[] stride, size_t[] padding, float p = 2.0f)
{
    Operation proj(Operation param)
    {
        auto norm = convParamsNorm(param, inShape, stride, padding, p);

        return maxNorm(param, norm, maxnorm);
    }

    return &proj;
}

Operation convParamsNorm(Operation param, size_t[] inShape, size_t[] stride, size_t[] padding, float p = 2.0f,
    size_t n = 2)
{
    if(p == 2.0f)
    {
        auto x = uniformSample([cast(size_t)1, param.shape[1]] ~ inShape) * 2.0f - 1.0f;

        for(int i = 0; i < n; i++)
        {
            x = x
            .convolution(param, padding, stride)
            .convolutionTranspose(param, padding, stride);
        }

        auto v = x / sqrt(sum(x * x));
        auto y = convolution(v, param, padding, stride);

        return sqrt(sum(y * y));
    }
    else if(p == 1.0f || p == float.infinity)
    {
        //Turns out this is equivalent, but only for $p \in \{1, infty\}$
        if(param.rank != 2)
        {
            param = param.reshape([param.shape[0], param.volume / param.shape[0]]);
        }

        return matrixNorm(param, p);
    }
    else
    {
        import std.conv : to;

        throw new Exception("Cannot compute convolution params norm for p=" ~ p.to!string);
    }
}

unittest
{
    auto k = float32([1, 1, 3, 3], [
        1, 2, 3, 4, 5, 6, 7, 8, 9
    ]);

    auto norm = convParamsNorm(k, [200, 200], [1, 1], [1, 1], 2.0f);

    import std.stdio;
    writeln(norm.evaluate().get!float[0]);
}

/**
    Performs a projection of param such that the new norm will be less than or equal to maxval.
*/
Operation maxNorm(Operation param, Operation norm, Operation maxval)
{
    return param * (1.0f / max(float32([], [1.0f]), norm / maxval));
}