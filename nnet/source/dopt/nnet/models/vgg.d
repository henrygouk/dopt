module dopt.nnet.models.vgg;

import std.math : isNaN;

import dopt.core;
import dopt.nnet;
import dopt.nnet.util;
import dopt.nnet.models.maybe;
import dopt.online : Projection;

class VGGOptions
{
    this()
    {
        _dropout = false;
        _batchnorm = false;
        _maxgainNorm = float.nan;
        _lipschitzNorm = float.nan;
        _maxNorm = float.infinity;
        _spectralDecay = 0.0f;
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
        "bool", "batchnorm",
        "float", "maxgainNorm",
        "float", "lipschitzNorm",
        "float", "maxNorm",
        "float", "spectralDecay"
    ));
}

Layer vgg16(Operation features, size_t[] denseLayerSizes = [4096, 4096], VGGOptions opts = new VGGOptions())
{
    auto sizes = [64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1];

    opts.verify();

    return makeExtractor(features, sizes, opts)
          .makeTop(denseLayerSizes, opts);
}

Layer vgg19(Operation features, size_t[] denseLayerSizes = [4096, 4096], VGGOptions opts = new VGGOptions())
{
    auto sizes = [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1];

    opts.verify();

    return makeExtractor(features, sizes, opts)
          .makeTop(denseLayerSizes, opts);
}

Layer vgg(Operation features, int[] extractorSizes, size_t[] denseLayerSizes = [4096, 4096],
    VGGOptions opts = new VGGOptions())
{
    opts.verify();

    return makeExtractor(features, extractorSizes, opts)
          .makeTop(denseLayerSizes, opts);
}

private
{
    Layer makeExtractor(Operation features, int[] sizes, VGGOptions opts)
    {
        auto layers = dataSource(features);
        int poolCtr;

        float drop = opts.dropout ? 0.2f : 0.0f;
        bool bn = opts.batchnorm;
        float bnlip = float.infinity;
        float maxgain = opts.maxNorm;

        if(!isNaN(opts.lipschitzNorm))
        {
            bnlip = opts.maxNorm;
        }
        
        if(isNaN(opts.maxgainNorm))
        {
            maxgain = float.infinity;
        }

        foreach(s; sizes)
        {
            if(s == -1)
            {
                layers = layers.maxPool([2, 2]);
                poolCtr++;
            }
            else
            {
                Projection projFunc;

                if(!isNaN(opts.lipschitzNorm))
                {
                    projFunc = projConvParams(
                        float32Constant(opts.maxNorm),
                        layers.trainOutput.shape[2 .. $],
                        [1, 1],
                        [1, 1],
                        opts.lipschitzNorm
                    );
                }

                layers = layers
                        .maybeDropout(poolCtr == 0 ? 0.0f : drop)
                        .conv2D(s, [3, 3], new Conv2DOptions()
                                              .padding([1, 1])
                                              .maxgain(maxgain)
                                              .filterProj(projFunc)
                                              .spectralDecay(opts.spectralDecay))
                        .maybeBatchNorm(bn, new BatchNormOptions()
                                               .maxgain(maxgain)
                                               .lipschitz(bnlip))
                        .relu();
            }
        }

        return layers;
    }

    Layer makeTop(Layer input, size_t[] sizes, VGGOptions opts)
    {
        float drop = opts.dropout ? 0.5f : 0.0f;

        Projection projFunc;
        float maxgain = opts.maxNorm;

        if(!isNaN(opts.lipschitzNorm))
        {
            projFunc = projMatrix(float32Constant(opts.maxNorm), opts.lipschitzNorm);
        }
        
        if(isNaN(opts.maxgainNorm))
        {
            maxgain = float.infinity;
        }

        foreach(i, s; sizes)
        {
            input = input
                   .maybeDropout(drop)
                   .dense(s, new DenseOptions()
                                .maxgain(maxgain)
                                .weightProj(projFunc)
                                .spectralDecay(opts.spectralDecay))
                   .relu();
        }

        return input;
    }
}
