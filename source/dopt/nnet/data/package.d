module dopt.nnet.data;

public
{
    import dopt.nnet.data.cifar10;
    import dopt.nnet.data.mnist;
}

struct Dataset
{
    float[][] trainFeatures;
    float[][] testFeatures;
    float[][] trainLabels;
    float[][] testLabels;
}