module dopt.nnet.models.maybe;

import dopt;

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

Layer maybeBatchNorm(Layer l, bool bn)
{
    if(bn)
    {
        return l.batchNorm();
    }
    else
    {
        return l;
    }
}