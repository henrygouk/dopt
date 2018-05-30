module dopt.nnet.models.wrn;

import std.math : isNaN;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.nnet.models.maybe;
import dopt.online;

class WRNOptions
{
    this()
    {
        _dropout = false;
        _maxgainNorm = float.nan;
        _lipschitzNorm = float.nan;
        _maxNorm = float.infinity;
        _spectralDecay = 0.0f;
        _weightDecay = 0.0001f;
        _stride = [1, 2, 2];
    }

    void verify()
    {
        import std.exception : enforce;

        int regCtr;

        if(!isNaN(_maxgainNorm))
        {
            regCtr++;

            enforce(_maxgainNorm == 2.0f, "Only a maxgainNorm of 2 is currently supported.");
        }

        if(!isNaN(_lipschitzNorm))
        {
            regCtr++;
        }

        enforce(regCtr <= 1, "VGG models currently only support using one of maxgain and the lipschitz constraint");
    }

    mixin(dynamicProperties(
        "bool", "dropout",
        "float", "maxgainNorm",
        "float", "lipschitzNorm",
        "float", "maxNorm",
        "float", "spectralDecay",
        "float", "weightDecay",
        "size_t[3]", "stride"
    ));
}

Layer wideResNet(Operation features, size_t depth, size_t width, WRNOptions opts = new WRNOptions())
{
    size_t n = (depth - 4) / 6;

    opts.verify();

    float maxgain = float.infinity;

    if(opts.maxgainNorm == 2.0f)
    {
        maxgain = opts.maxNorm;
    }

    float lambda = float.infinity;
    float lipschitzNorm = float.nan;

    if(!isNaN(opts.lipschitzNorm))
    {
        lipschitzNorm = opts.lipschitzNorm;
        lambda = opts.maxNorm;
    }

    Projection filterProj;
    Operation lambdaSym = float32Constant(lambda);

    if(lambda != float.infinity)
    {
        filterProj = projConvParams(lambdaSym, features.shape[2 .. $], [1, 1], [1, 1], lipschitzNorm);
    }

    auto pred = dataSource(features)
               .conv2D(16, [3, 3], new Conv2DOptions()
                    .padding([1, 1])
                    .useBias(false)
                    .weightDecay(opts.weightDecay)
                    .maxgain(maxgain)
                    .filterProj(filterProj))
               .wrnBlock(16 * width, n, opts.stride[0], opts)
               .wrnBlock(32 * width, n, opts.stride[1], opts)
               .wrnBlock(64 * width, n, opts.stride[2], opts)
               .batchNorm(new BatchNormOptions().maxgain(maxgain))
               .relu()
               .meanPool();

    return pred;
}

private Layer wrnBlock(Layer inLayer, size_t u, size_t n, size_t s, WRNOptions opts)
{
    float maxgain = float.infinity;

    if(opts.maxgainNorm == 2.0f)
    {
        maxgain = opts.maxNorm;
    }

    float lambda = float.infinity;
    float lipschitzNorm = float.nan;

    if(!isNaN(opts.lipschitzNorm))
    {
        lipschitzNorm = opts.lipschitzNorm;
        lambda = opts.maxNorm;
    }

    Operation lambdaSym = float32Constant(lambda);

    auto convOpts()
    {
        return new Conv2DOptions()
            .padding([1, 1])
            .useBias(false)
            .weightDecay(opts.weightDecay)
            .maxgain(maxgain);
    }

    auto bnOpts()
    {
        return new BatchNormOptions()
            .maxgain(maxgain)
            .lipschitz(lambda);
    }

    Layer res;

    for(size_t i = 0; i < n; i++)
    {
        res = inLayer
            .batchNorm(bnOpts())
            .relu();
        
        Projection filterProj = null;

        if(lambda != float.infinity)
        {
            filterProj = projConvParams(lambdaSym, res.trainOutput.shape[2 .. $], [s, s], [1, 1], lipschitzNorm);
        }

        res = res
             .conv2D(u, [3, 3], convOpts().stride([s, s]).filterProj(filterProj))
             .batchNorm(bnOpts())
             .relu()
             .maybeDropout(opts.dropout ? 0.3f : 0.0f);
        
        if(lambda != float.infinity)
        {
            filterProj = projConvParams(lambdaSym, res.trainOutput.shape[2 .. $], [1, 1], [1, 1], lipschitzNorm);
        }
        
        res = res
             .conv2D(u, [3, 3], convOpts().filterProj(filterProj));
        
        Layer shortcut = inLayer;
        
        if(inLayer.output.shape[1] != res.output.shape[1])
        {
            if(lambda != float.infinity)
            {
                filterProj = projConvParams(lambdaSym, inLayer.trainOutput.shape[2 .. $], [s, s], [1, 1],
                    lipschitzNorm);
            }

            shortcut = inLayer.conv2D(u, [1, 1], new Conv2DOptions()
                                                .stride([s, s])
                                                .useBias(false)
                                                .weightDecay(opts.weightDecay)
                                                .maxgain(maxgain)
                                                .filterProj(filterProj));
        }

        res = new Layer(
            [res, shortcut],
            res.output + shortcut.output,
            res.trainOutput + shortcut.trainOutput,
            []
        );

        inLayer = res;
        s = 1;
    }

    return res;
}

private Layer meanPool(Layer input)
{
    Operation meanPoolImpl(Operation inp)
    {
        auto mapVol = inp.shape[2] * inp.shape[3];
        float scale = 1.0f / mapVol;

        return inp.reshape([inp.shape[0] * inp.shape[1], inp.shape[2] * inp.shape[3]])
                  .sum([1])
                  .reshape([inp.shape[0], inp.shape[1]]) * scale;
    }

    auto y = meanPoolImpl(input.output);
    auto yTr = meanPoolImpl(input.trainOutput);

    return new Layer([input], y, yTr, []);
}
