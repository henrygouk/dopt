module dopt.nnet.models.vgg;

import dopt.core;
import dopt.nnet;
import dopt.nnet.models.maybe;

Layer vgg16(Operation features, size_t[] denseLayerSizes = [4096, 4096], bool drop = true, bool bn = true,
    float maxgain = 3.0f)
{
    auto sizes = [64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1];

    return makeExtractor(features, sizes, drop ? 0.2f : 0.0f, bn, maxgain)
          .makeTop(denseLayerSizes, drop ? 0.5f : 0.0f, bn, maxgain);
}

Layer vgg19(Operation features, size_t[] denseLayerSizes = [4096, 4096], bool drop = true, bool bn = true,
	float maxgain = 3.0f)
{
    auto sizes = [64, 64, -1, 128, 128, -1, 256, 256, 256, 256, -1, 512, 512, 512, 512, -1, 512, 512, 512, 512, -1];

    return makeExtractor(features, sizes, drop ? 0.2f : 0.0f, bn, maxgain)
          .makeTop(denseLayerSizes, drop ? 0.5f : 0.0f, bn, maxgain);
}

private
{
    Layer makeExtractor(Operation features, int[] sizes, float drop, bool bn, float maxgain)
    {
        auto layers = dataSource(features);
        int poolCtr;

        foreach(s; sizes)
        {
            if(s == -1)
            {
                layers = layers.maxPool([2, 2]);
                poolCtr++;
            }
            else
            {
                layers = layers
                        .maybeDropout(poolCtr == 0 ? 0.0f : drop)
                        .conv2D(s, [3, 3], new Conv2DOptions().padding([1, 1]).maxgain(maxgain))
                        .maybeBatchNorm(bn, new BatchNormOptions().maxgain(maxgain))
                        .relu();
            }
        }

        return layers;
    }

    Layer makeTop(Layer input, size_t[] sizes, float drop, bool bn, float maxgain)
    {
        foreach(i, s; sizes)
        {
            input = input
                   .maybeDropout(drop)
                   .dense(s, new DenseOptions().maxgain(maxgain))
                   .relu();
        }

        return input;
    }
}
