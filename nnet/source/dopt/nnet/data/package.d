module dopt.nnet.data;

public
{
    import dopt.nnet.data.cifar;
    import dopt.nnet.data.imagetransformer;
    import dopt.nnet.data.mnist;
    import dopt.nnet.data.sins;
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

/**
    Classification or Regression Dataset.

    This dataset provider will store an entire classification or regression dataset in memory. It assumes that each
    object in the dataset is composed of a feature vector and a label vector.
*/
class CORDataset : Dataset
{
    public
    {
        this(float[][][] features, float[][][] labels, size_t[] shape)
        {
            import std.algorithm : fold, map;
            import std.array : array;

            enforce(features.length == labels.length, "features.length != labels.length");
            enforce(shape.length != 0, "shape.length == 0");

            mFeatures = features.map!(x => x.dup).array();
            mLabels = labels.map!(x => x.dup).array();
            mShape = shape.dup;
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
            enforce(foldIdx < mFeatures.length, "Invalid foldIdx");

            return mFeatures[foldIdx].length;
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

            enforce(foldIdx < mFeatures.length, "Invalid foldIdx");
            enforce(batchData.length == 2, "batchData.length != 2");

            return getBatchImpl(mFeatures[foldIdx], mLabels[foldIdx]);
        }

        void shuffle(size_t foldIdx)
        {
            import std.random : randomShuffle;
            import std.range : zip;

            enforce(foldIdx < mFeatures.length, "Invalid foldIdx");

            randomShuffle(zip(mFeatures[foldIdx], mLabels[foldIdx]));
        }
    }

    protected
    {
        float[][][] mFeatures;
        float[][][] mLabels;
        size_t[] mShape;
        size_t mVolume;
    }
}