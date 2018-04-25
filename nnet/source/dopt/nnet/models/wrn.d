module dopt.nnet.models.wrn;

import dopt.core;
import dopt.nnet;
import dopt.nnet.models.maybe;

Layer wideResNet(Operation features, size_t depth, size_t width, size_t[3] stride = [1, 2, 2], bool drop = true,
    float maxgain = float.infinity)
{
    size_t n = (depth - 4) / 6;

    auto pred = dataSource(features)
               .conv2D(16, [3, 3], new Conv2DOptions()
                    .padding([1, 1])
                    .useBias(false)
                    .weightDecay(0.0001f)
                    .maxgain(maxgain))
               .wrnBlock(16 * width, n, stride[0], drop, maxgain)
               .wrnBlock(32 * width, n, stride[1], drop, maxgain)
               .wrnBlock(64 * width, n, stride[2], drop, maxgain)
               .batchNorm(new BatchNormOptions().maxgain(maxgain))
               .relu()
               .meanPool();

    return pred;
}

private Layer wrnBlock(Layer inLayer, size_t u, size_t n, size_t s, bool drop, float maxgain)
{
    auto convOpts()
    {
        return new Conv2DOptions()
            .padding([1, 1])
            .useBias(false)
            .weightDecay(0.0001f)
            .maxgain(maxgain);
    }

    auto bnOpts()
    {
        return new BatchNormOptions()
            .maxgain(maxgain);
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
            .maybeDropout(drop ? 0.3f : 0.0f)
            .conv2D(u, [3, 3], convOpts());
        
        Layer shortcut = inLayer;
        
        if(inLayer.output.shape[1] != res.output.shape[1])
        {
            shortcut = inLayer.conv2D(u, [1, 1], new Conv2DOptions()
                                                .stride([s, s])
                                                .useBias(false)
                                                .weightDecay(0.0001f)
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