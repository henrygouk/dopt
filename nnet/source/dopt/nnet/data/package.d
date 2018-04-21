module dopt.nnet.data;

public
{
    import dopt.nnet.data.cifar;
    import dopt.nnet.data.imagetransformer;
    import dopt.nnet.data.mnist;
}

import std.exception : enforce;

interface Dataset
{
    size_t[] shape();
    size_t volume();
    size_t foldSize(size_t foldIdx);
    size_t getBatch(float[][] batchData, size_t batchIdx, size_t foldIdx);
    void shuffle(size_t foldIdx);
}

class HoldOutDataset : Dataset
{
    public
    {
        this(float[][] trainFeatures, float[][] testFeatures, float[][] trainLabels, float[][] testLabels,
            size_t[] shape)
        {
            import std.algorithm : fold;

            enforce(trainFeatures.length == trainLabels.length, "trainFeatures.length != trainLabels.length");
            enforce(testFeatures.length == testLabels.length, "testFeatures.length != testLabels.length");
            enforce(shape.length != 0, "shape.length == 0");

            mTrainFeatures = trainFeatures;
            mTestFeatures = testFeatures;
            mTrainLabels = trainLabels;
            mTestLabels = testLabels;
            mShape = shape;
            mVolume = shape.fold!((a, b) => a * b);
        }

        size_t[] shape()
        {
            return mShape;
        }

        size_t volume()
        {
            return mVolume;
        }

        size_t foldSize(size_t foldIdx)
        {
            enforce(foldIdx == 0 || foldIdx == 1, "foldIdx must be 0 (train) or 1 (test)");

            if(foldIdx == 0)
            {
                return mTrainFeatures.length;
            }
            else
            {
                return mTestFeatures.length;
            }
        }

        size_t getBatch(float[][] batchData, size_t batchIdx, size_t foldIdx)
        {
            import std.algorithm : copy, joiner;
            import std.range : chunks;

            size_t getBatchImpl(float[][] fs, float[][] ls)
            {
                size_t batchSize = batchData[0].length / mVolume;

                auto fsChunks = fs.chunks(batchSize);
                auto rem = fsChunks[batchIdx]
                          .joiner()
                          .copy(batchData[0]);
                
                rem[] = 0.0f;

                rem = ls.chunks(batchSize)[batchIdx]
                        .joiner()
                        .copy(batchData[1]);
                
                rem[] = 0.0f;

                return (batchIdx + 1) % fsChunks.length;
            }

            enforce(foldIdx == 0 || foldIdx == 1, "foldIdx must be 0 (train) or 1 (test)");
            enforce(batchData.length == 2, "batchData.length != 2");

            if(foldIdx == 0)
            {
                return getBatchImpl(mTrainFeatures, mTrainLabels);
            }
            else
            {
                return getBatchImpl(mTestFeatures, mTestLabels);
            }
        }

        void shuffle(size_t foldIdx)
        {
            import std.random : randomShuffle;
            import std.range : zip;

            enforce(foldIdx == 0 || foldIdx == 1, "foldIdx must be 0 (train) or 1 (test)");

            if(foldIdx == 0)
            {
                randomShuffle(zip(mTrainFeatures, mTrainLabels));
            }
            else
            {
                randomShuffle(zip(mTestFeatures, mTestLabels));
            }
        }
    }

    protected
    {
        float[][] mTrainFeatures;
        float[][] mTestFeatures;
        float[][] mTrainLabels;
        float[][] mTestLabels;
        size_t[] mShape;
        size_t mVolume;
    }
}