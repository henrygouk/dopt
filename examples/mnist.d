#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
+/
module mnist;

/*
	This example trains a small convolutional network on the MNIST dataset of hand-written images. The network used is
	very small by today's standards, but MNIST is a very easy dataset so this does not really matter.

	The MNIST dataset contains small monochrome images of hand-written digits, and the goal is to determine which digit
	each image contains.
*/
void main(string[] args)
{
	import std.algorithm : joiner, maxIndex;
	import std.array : array;
	import std.range : zip, chunks;
	import std.stdio : stderr, writeln;
	
	import dopt.core;
	import dopt.nnet;
	import dopt.online;

	if(args.length != 2)
	{
		stderr.writeln("Usage: mnist.d <data directory>");
		return;
	}

	//Load the minst dataset of hand-written digits. Download the binary files from http://yann.lecun.com/exdb/mnist/
    auto data = loadMNIST(args[1]);

	//Create the variables nodes required to pass data into the operation graph
	size_t batchSize = 100;
    auto features = float32([batchSize, 1, 28, 28]);
	auto labels = float32([batchSize, 10]);

	//Construct a small convolutional network
	auto preds = dataSource(features)
				.conv2D(32, [5, 5])
				.relu()
				.maxPool([2, 2])
				.conv2D(32, [5, 5])
				.relu()
				.maxPool([2, 2])
				.dense(10)
				.softmax();

	//Construct the DAGNetwork object that can be used to collate all the parameters and loss terms
	auto network = new DAGNetwork([features], [preds]);

	//Create a symbol to represent the training loss function
	auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;

	//Create an optimiser that can use minibatches of labelled data to update the weights of the network
	auto lr = float32([], [0.001f]);
	auto updater = adam([lossSym], network.params, network.paramProj);

	//Iterate for 10 epochs of training
	foreach(e; 0 .. 40)
	{
		float totloss = 0;
		float tot = 0;

		if(e == 30)
		{
			lr.value.as!float[0] = 0.0001f;
		}

		//Iterate over each minibatch of data and perform an update of the model parameters
		foreach(fs, ls; zip(data.trainFeatures.chunks(batchSize), data.trainLabels.chunks(batchSize)))
		{
			auto loss = updater([
				features: Buffer(fs.joiner().array()),
				labels: Buffer(ls.joiner().array())
			]);

			totloss += loss[0].as!float[0];
			tot++;
		}

		//Write out the training loss for this epoch
		writeln(e, ": ", totloss / tot);
	}

	int correct;
	int total;

	import std.stdio : writeln;

	//Iterate over each minibatch in the test set
	foreach(fs, ls; zip(data.testFeatures.chunks(batchSize), data.testLabels.chunks(batchSize)))
	{
		//Make some predictions for this minibatch
		auto pred = network.outputs[0].evaluate([
			features: Buffer(fs.joiner().array())
		]).as!float;

		//Determine the accuracy of these predictions using the ground truth data
		foreach(p, t; zip(pred.chunks(10), ls))
		{
			if(p.maxIndex == t.maxIndex)
			{
				correct++;
			}

			total++;
		}
	}

	//Write out the accuracy of the model on the test set
	writeln(correct / cast(float)total);
}