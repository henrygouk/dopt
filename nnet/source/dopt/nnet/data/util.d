module dopt.nnet.data.util;

struct Dataset
{
    float[][] trainFeatures;
    float[][] testFeatures;
    float[][] trainLabels;
    float[][] testLabels;
}

void standardise(Dataset data)
{
    auto mean = new float[data.trainFeatures[0].length];
    auto var = new float[data.trainFeatures[0].length];
    mean[] = 0;
    var[] = 0;

    foreach(x; data.trainFeatures)
    {
        mean[] += x[];
    }

    mean[] /= data.trainFeatures.length;

    foreach(x; data.trainFeatures)
    {
        var[] += (mean[] - x[]) * (mean[] - x[]);
    }

    var[] /= data.trainFeatures.length;

    for(size_t i = 0; i < var.length; i++)
    {
        import std.math : sqrt;

        var[i] = 1.0f / sqrt(var[i]);
    }

    import std.range : chain;

    foreach(x; chain(data.trainFeatures, data.testFeatures))
    {
        x[] -= mean[];
        x[] *= var[];
    }
}