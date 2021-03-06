module dopt.nnet.models.maybe;

import dopt.nnet;

Layer maybeDropout(Layer l, float prob)
{
    if(prob != 0)
    {
        return l.dropout(prob);
    }
    else
    {
        return l;
    }
}

Layer maybeBatchNorm(Layer l, bool bn, BatchNormOptions opts = new BatchNormOptions())
{
    if(bn)
    {
        return l.batchNorm(opts);
    }
    else
    {
        return l;
    }
}