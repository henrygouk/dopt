/**
    This package contains a deep learning API backed by dopt.

    Working examples for how this package can be used are given in the $(D examples/mnist.d) and $(D examples/cifar10.d)
    files.

    One would generally start by using UFCS to define a feed-forward network:

    ---
    auto features = float32([128, 1, 28, 28]);

    auto layers = dataSource(features)
                 .dense(2_000)
                 .relu()
                 .dense(2_000)
                 .relu()
                 .dense(10)
                 .softmax();
    ---

    The $(D DAGNetwork) class can then be used to traverse the resulting graph and aggregate parameters/loss terms:

    ---
    auto network = new DAGNetwork([features], layers);
    ---

    After this, one can define an objective function---there are a few standard loss functions implemented in
    $(D dopt.nnet.losses):

    ---
    auto labels = float32([128, 10]);

    auto trainLoss = crossEntropy(layers.trainOutput, labels) + network.paramLoss;
    ---

    where `network.paramLoss` is the sum of any parameter regularisation terms. The $(D dopt.online) package can be
    used to construct an updater:

    ---
    auto updater = sgd([trainLoss], network.params, network.paramProj);
    ---

    Finally, one can call this updater with some actual training data:

    ---
    updater([
        features: Buffer(some_real_features),
        labels: Buffer(some_real_labels)
    ]);
    ---

    Authors: Henry Gouk
*/
module dopt.nnet;

public
{
    import dopt.nnet.data;
    import dopt.nnet.layers;
    import dopt.nnet.losses;
    import dopt.nnet.models;
    import dopt.nnet.networks;
    import dopt.nnet.parameters;
}

version(Have_DoptCUDA)
{
    import dopt.cuda;
}