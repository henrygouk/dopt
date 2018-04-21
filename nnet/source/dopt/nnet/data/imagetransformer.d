module dopt.nnet.data.imagetransformer;

import dopt.nnet.data;

class ImageTransformer : Dataset
{
    public
    {
        this(Dataset imageDataset, size_t jitterX, size_t jitterY, bool flipX, bool flipY, size_t[] folds = [0])
        {
            mDataset = imageDataset;
            mJitterX = jitterX;
            mJitterY = jitterY;
            mFlipX = flipX;
            mFlipY = flipY;
            mFolds = folds.dup;
            mPadded = new float[shape[0] * (shape[1] + 2 * jitterY) * (shape[2] + 2 * jitterX)];
        }

        size_t[] shape()
        {
            return mDataset.shape();
        }

        size_t volume()
        {
            return mDataset.volume();
        }

        size_t foldSize(size_t foldIdx)
        {
            return mDataset.foldSize(foldIdx);
        }

        size_t getBatch(float[][] batchData, size_t batchIdx, size_t foldIdx)
        {
            import std.algorithm : canFind, reverse;
            import std.random : uniform;
            import std.range : chunks, drop, stride, take;

            auto ret = mDataset.getBatch(batchData, batchIdx, foldIdx);

            size_t pw = shape[2] + 2 * mJitterX;
            size_t ph = shape[1] + 2 * mJitterY;
            
            if(mFolds.canFind(foldIdx))
            {
                foreach(img; batchData[0].chunks(volume))
                {
                    if(mJitterX != 0 || mJitterY != 0)
                    {
                        //Pad the image. The extra content around the border will be filled with reflected image.
                        for(size_t c = 0; c < shape[0]; c++)
                        {
                            for(size_t y = 0; y < shape[1]; y++)
                            {
                                for(size_t x = 0; x < shape[2]; x++)
                                {
                                    mPadded[c * ph * pw + (y + mJitterY) * pw + x + mJitterX] =
                                        img[c * shape[1] * shape[2] + y * shape[2] + x];
                                }

                                if(mJitterX != 0)
                                {
                                    size_t o = c * ph * pw + (y + mJitterY) * pw;
                                    mPadded[o .. o + mJitterX] = mPadded[o + mJitterX .. o + 2 * mJitterX];
                                    mPadded[o .. o + mJitterX].reverse();

                                    o += shape[2];
                                    mPadded[o + mJitterX .. o + 2 * mJitterX] = mPadded[o .. o + mJitterX];
                                    mPadded[o + mJitterX .. o + 2 * mJitterX].reverse();
                                }
                            }

                            for(size_t y = 0; y < mJitterY; y++)
                            {
                                size_t o = c * pw * ph;

                                //Pad the top rows
                                mPadded[o + y * pw .. o + (y + 1) * pw] =
                                    mPadded[o + (2 * mJitterY - y - 1) * pw .. o + (2 * mJitterY - y) * pw];
                                
                                //Pad the bottom rows
                                mPadded[o + (ph - y - 1) * pw .. o + (ph - y) * pw] =
                                    mPadded[o + (ph - 2 * mJitterY + y) * pw .. o + (ph - 2 * mJitterY + y + 1) * pw];
                            }
                        }

                        size_t xOff = uniform(0, mJitterX * 2);
                        size_t yOff = uniform(0, mJitterY * 2);

                        //Crop the padded image
                        for(size_t c = 0; c < shape[0]; c++)
                        {
                            for(size_t y = 0; y < shape[1]; y++)
                            {
                                for(size_t x = 0; x < shape[2]; x++)
                                {
                                    img[c * shape[1] * shape[2] + y * shape[2] + x] =
                                        mPadded[c * ph * pw + (y + yOff) * pw + x + xOff];
                                }
                            }
                        }
                    }

                    if(mFlipX && uniform(0.0f, 1.0f) < 0.5f)
                    {
                        foreach(row; img.chunks(mDataset.shape[2]))
                        {
                            row.reverse();
                        }
                    }

                    if(mFlipY && uniform(0.0f, 1.0f) < 0.5f)
                    {
                        for(size_t c = 0; c < shape[0]; c++)
                        {
                            for(size_t x = 0; x < shape[2]; x++)
                            {
                                img.drop(c * shape[1] * shape[2] + x)
                                   .stride(shape[2])
                                   .take(shape[1])
                                   .reverse();
                            }
                        }
                    }
                }
            }

            return ret;
        }

        void shuffle(size_t foldIdx)
        {
            mDataset.shuffle(foldIdx);
        }
    }

    protected
    {
        Dataset mDataset;
        size_t mJitterX;
        size_t mJitterY;
        bool mFlipX;
        bool mFlipY;
        size_t[] mFolds;
        float[] mPadded;
    }
}