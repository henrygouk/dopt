module dopt.nnet.data;

public
{
    import dopt.nnet.data.cifar;
    import dopt.nnet.data.imagetransformer;
    import dopt.nnet.data.mnist;
    import dopt.nnet.data.sins;
}

import std.exception : enforce;

interface BatchIterator
{
    size_t[][] shape();
    size_t[] volume();
    size_t length();
    void getBatch(float[][] batchData);
    bool finished();
    void restart();
}

/**
    A $(D BatchIterator) specialisation for supervised learning tasks.
*/
class SupervisedBatchIterator : BatchIterator
{
    public
    {
        this(float[][] features, float[][] labels, size_t[][] shape, bool shuffle)
        {
            import std.algorithm : fold, map;
            import std.array : array;
            import std.range : iota;

            enforce(features.length == labels.length, "features.length != labels.length");
            enforce(shape.length != 0, "shape.length == 0");

            mFeatures = features.dup;
            mLabels = labels.dup;
            mShape = shape.map!(x => x.dup).array;
            mShuffle = shuffle;
            mVolumes = shape
                      .map!(y => y.fold!((a, b) => a * b))
                      .array();

            mIndices = iota(0, mFeatures.length).array();
        }

        size_t[][] shape()
        {
            return mShape;
        }

        size_t[] volume()
        {
            return mVolumes;
        }

        size_t length()
        {
            return mFeatures.length;
        }

        bool finished()
        {
            return mFront >= length;
        }

        void restart()
        {
            import std.random : randomShuffle;

            if(mShuffle)
            {
                mIndices.randomShuffle();
            }

            mFront = 0;
        }

        void getBatch(float[][] batchData)
        {
            import std.algorithm : map, joiner, copy;
            import std.range : drop, take;

            //Check the size of the arguments
            enforce(batchData.length == 2, "SupervisedBatchIterator.getBatch expects two arrays to fill.");
            enforce(batchData[0].length % volume[0] == 0, "batchData[0].length % volume[0] != 0");
            enforce(batchData[1].length % volume[1] == 0, "batchData[1].length % volume[1] != 0");
            enforce(batchData[0].length / volume[0] == batchData[1].length / volume[1],
                "batchData[0].length / volume[0] != batchData[1].length / volume[1]");
            
            size_t batchSize = batchData[0].length / volume[0];

            batchData[0][] = 0;
            batchData[1][] = 0;

            mIndices.drop(mFront)
                    .take(batchSize)
                    .map!(x => mFeatures[x])
                    .joiner()
                    .copy(batchData[0]);
            
            mIndices.drop(mFront)
                    .take(batchSize)
                    .map!(x => mLabels[x])
                    .joiner()
                    .copy(batchData[1]);

            mFront += batchSize;
        }
    }

    protected
    {
        float[][] mFeatures;
        float[][] mLabels;
        size_t[][] mShape;
        size_t[] mVolumes;
        bool mShuffle;
        size_t mFront;
        size_t[] mIndices;
    }
}