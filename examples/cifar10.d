#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
+/
module cifar10;

/*
	This example trains a VGG19-style network on the CIFAR-10 dataset of tiny images.

	VGG networks are fairly easy to understand, compared to some of the more recently presented models like GoogLeNet.
	See ``Very Deep Convolutional Networks for Large-Scale Image Recognition'' by Simonyan and Zisserman for more
	details. This example uses the dopt.nnet.models package to make defining a VGG model very easy.

	The CIFAR-10 dataset contains 60,000 32x32 pixel colour images. Each of these images belongs to one of 10 classes.
	In the standard setting, 50,000 of these images are used for training a model, and the other 10,000 are used for
	evaluating how well the model works.
*/
void main(string[] args)
{
	import std.algorithm : joiner, maxIndex;
	import std.array : array;
	import std.range : zip, chunks;
	import std.stdio : stderr, stdout, write, writeln;

	import dopt.core;
	import dopt.nnet;
	import dopt.online;

	if(args.length != 2)
	{
		stderr.writeln("Usage: cifar10.d <data directory>");
		return;
	}

	//Loads the CIFAR-10 dataset. Download this in the binary format from https://www.cs.toronto.edu/~kriz/cifar.html
	writeln("Loading data...");
    auto data = loadCIFAR10(args[1]);

	/*
	Now we create two variable nodes. ``features'' is used to represent a minibatch of input images, and ``labels''
	will be used to represent the label corresponding to each of those images.
	*/
	writeln("Constructing network graph...");
	size_t batchSize = 100;
    auto features = float32([batchSize, 3, 32, 32]);
    auto labels = float32([batchSize, 10]);

	/*
	There are a few predefined models in dopt.nnet.models, such as vgg19. We provide it with the variable we want to
	use as the input to this model, tell it what sizes the fully connected layers should be, and then put a softmax
	activation function on the end. The softmax function is the standard activation function when one is performing
	a classification task.
	*/
    auto preds = vgg19(features, [512, 512])
				.dense(10)
				.softmax();
    
	//The DAGNetwork class takes the inputs and outputs of a network and aggregates parameters in several different.
    auto network = new DAGNetwork([features], [preds]);

	/*
	Layer objects have both ``output'' and ``trainOutput'' fields, because operations like dropout perform different
	computations at train and test time. Therefore, we construct two different loss symbols: one for optimising, and
	one for evaluating.
	*/
    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLoss = crossEntropy(preds.output, labels);

	/*
	Now we set up an optimiser. Adam is good for proof of concepts, due to the fast convergence, however the
	performance of the final model is often slightly worse than that of a model trained with SGD+momentum.
	*/
	writeln("Creating optimiser...");
	auto learningRate = float32([], [0.0001f]);
	auto updater = adam([lossSym], network.params, network.paramProj, learningRate);

	writeln("Training...");

	//Iterate for 120 epochs of training!
	foreach(e; 0 .. 120)
	{
		float tot = 0;

		//Decreasing the learning rate after a while often results in better performance.
		if(e == 100)
		{
			learningRate.value.as!float[0] = 0.00001f;
		}

		//Iterate through all the training data, creating minibatches as we go.
		foreach(fs, ls; zip(data.trainFeatures.chunks(batchSize), data.trainLabels.chunks(batchSize)))
		{
			//Make an update to the model parameters using the minibatch of training data
			updater([
				features: Buffer(fs.joiner().array()),
				labels: Buffer(ls.joiner().array())
			]);

			tot++;

			//Let the user know that we are making some progress
            write("  ", tot, "/500    \r");
            stdout.flush();
		}

		int correct;
		int total;

		/*
		After an epoch of training, show how well the model is performing on the test set---this could be done less
		frequently, or on a validation set if one is still tuning the structure/hyperparams of the model.
		*/
		foreach(fs, ls; zip(data.testFeatures.chunks(batchSize), data.testLabels.chunks(batchSize)))
		{
			auto pred = network.outputs[0].evaluate([
				features: Buffer(fs.joiner().array())
			]).as!float;

			foreach(p, t; zip(pred.chunks(10), ls))
			{
				if(p.maxIndex == t.maxIndex)
				{
					correct++;
				}

				total++;
			}
		}

		//Tell the user how accurate the model is on the test data
		writeln(e + 1, ": ", correct / cast(float)total);
	}
}