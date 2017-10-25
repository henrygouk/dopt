/**
    Authors: Henry Gouk
*/
module dopt.nnet.losses;

import dopt.core;

/**
    Creates a cross entropy loss term suitable for multiclass classification problems.

    It is assumed that the two input operations are rank-2 tensors, where the first dimension is an index into the
    batch, and the second index is the index into the label probabilities.

    Params:
        hypothesis = The predictions made by a model.
        groundTruth = The true values for the labels, as provided by the training dataset.
    
    Returns:
        An $(D Operation) representing the mean cross entropy loss.
*/
Operation crossEntropy(Operation hypothesis, Operation groundTruth)
{
    return sum(groundTruth * log(hypothesis + 1e-6f)) * (-1.0f / hypothesis.shape[0]);
}

/**
    Creates a squared error loss term suitable for regression (and multi-target regression) problems.

    Params:
        hypothesis = The predictions made by the model.
        groundTruth = The true values for the targets, as provided by the training dataset.
*/
Operation squaredError(Operation hypothesis, Operation groundTruth)
{
    auto diff = hypothesis - groundTruth;

    return sum(diff * diff) * (1.0f / hypothesis.shape[0]);
}