module dopt.nnet.models.wrn;

import std.math : isNaN;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.nnet.models.maybe;

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

    if(!isNaN(opts.maxgainNorm))
    {
        maxgain = opts.maxNorm;
    }

    auto pred = dataSource(features)
               .conv2D(16, [3, 3], new Conv2DOptions()
                    .padding([1, 1])
                    .useBias(false)
                    .weightDecay(opts.weightDecay)
                    .maxgain(maxgain))
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

    if(!isNaN(opts.maxgainNorm))
    {
        maxgain = opts.maxNorm;
    }

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
        float bnlip = !isNaN(opts.lipschitzNorm) ? opts.maxNorm : float.infinity;

        return new BatchNormOptions()
            .maxgain(maxgain)
            .lipschitz(bnlip);
    }

    Layer res;

    for(size_t i = 0; i < n; i++)
    {
        res = inLayer
            .batchNorm(bnOpts())
            .relu()
            .conv2D(u, [3, 3], convOpts().stride([s, s]))
            .batchNorm(bnOpts())
            .relu()
            .maybeDropout(opts.dropout ? 0.3f : 0.0f)
            .conv2D(u, [3, 3], convOpts());
        
        Layer shortcut = inLayer;
        
        if(inLayer.output.shape[1] != res.output.shape[1])
        {
            shortcut = inLayer.conv2D(u, [1, 1], new Conv2DOptions()
                                                .stride([s, s])
                                                .useBias(false)
                                                .weightDecay(opts.weightDecay)
                                                .maxgain(maxgain));
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