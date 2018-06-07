module dopt.nnet.data.svhn;

import std.algorithm;
import std.array;
import std.file;
import std.range;
import std.typecons;

import dopt.nnet.data;

auto loadSVHN(string directory, bool validation = false)
{
    auto loadFeatures(string filename)
    {
        return (cast(ubyte[])read(directory ~ "/" ~ filename))
              .map!(x => cast(float)x / 128.0f - 1.0f)
              .chunks(32 * 32 * 3)
              .map!(x => x.array())
              .array();
    }

    auto loadLabels(string filename)
    {
        auto lbls = (cast(ubyte[])read(directory ~ "/" ~ filename));

        auto ret = new float[][lbls.length];

        for(size_t i = 0; i < lbls.length; i++)
        {
            ret[i] = new float[10];
            ret[i][] = 0.0f;
            ret[i][lbls[i] - 1] = 1.0f;
        }

        return ret;
    }

    auto trainFeatures = loadFeatures("train_X.bin") ~ loadFeatures("extra_X.bin");
    auto testFeatures = loadFeatures("test_X.bin");
    auto trainLabels = loadLabels("train_y.bin") ~ loadLabels("extra_y.bin");
    auto testLabels = loadLabels("test_y.bin");

    if(validation)
    {
        testFeatures = trainFeatures[0 .. 10_000];
        testLabels = trainLabels[0 .. 10_000];
        trainFeatures = trainFeatures[10_000 .. $];
        trainLabels = trainLabels[10_000 .. $];
    }

    BatchIterator trainData = new SupervisedBatchIterator(
        trainFeatures,
        trainLabels,
        [[cast(size_t)3, 32, 32], [cast(size_t)10]],
        true
    );

    BatchIterator testData = new SupervisedBatchIterator(
        testFeatures,
        testLabels,
        [[cast(size_t)3, 32, 32], [cast(size_t)10]],
        false
    );

    return tuple!("train", "test")(trainData, testData);
}
