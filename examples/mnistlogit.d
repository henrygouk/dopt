#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
+/
module mnistlogit;

/*
	This example trains a logistic regression model on the MNIST dataset of hand-written images.

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
    auto features = float32([batchSize, 28 * 28]);
	auto labels = float32([batchSize, 10]);

	//Construct a logistic regression model
    auto W = float32([28 * 28, 10]);
    auto b = float32([10]);
    auto linComb = matmul(features, W) + b.repeat(batchSize);
	auto numerator = exp(linComb - linComb.maxElement());
    auto denominator = numerator.sum([1]).reshape([batchSize, 1]).repeat([1, 10]);
    auto preds = numerator / denominator;

	//Create a symbol to represent the training loss function
	auto lossSym = crossEntropy(preds, labels);

	//Create an optimiser that can use minibatches of labelled data to update the parameters of the model
	auto lr = float32([], [0.001f]);
	auto updater = adam([lossSym], [W, b], null, lr);
	auto testPlan = compile([preds]);

	size_t bidx;
	float[] fs = new float[features.volume];
	float[] ls = new float[labels.volume];

	//Iterate for 50 epochs of training
	foreach(e; 0 .. 50)
	{
		float totloss = 0;
		float tot = 0;

		data.train.restart();

		while(!data.train.finished())
		{
			//Get the next batch of training data (put into [fs, ls]). Update bidx with the next batch index.
			data.train.getBatch([fs, ls]);

			auto loss = updater([
				features: Buffer(fs),
				labels: Buffer(ls)
			]);

			totloss += loss[0].get!float[0];
			tot++;
		}

		//Write out the training loss for this epoch
		writeln(e, ": ", totloss / tot);
	}

	int correct;
	int total;

	import std.stdio : writeln;

	while(!data.test.finished)
	{
		//Get the next batch of test data (put into [fs, ls]). Update bidx with the next batch index.
		data.test.getBatch([fs, ls]);

		//Make some predictions for this minibatch
		auto pred = testPlan.execute([
			features: Buffer(fs)
		])[0].get!float;

		//Determine the accuracy of these predictions using the ground truth data
		foreach(p, t; zip(pred.chunks(10), ls.chunks(10)))
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