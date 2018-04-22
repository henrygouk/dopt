#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
dependency "progress-d" version="~>1.0.0"
+/
module cifar10;

import dopt.core;
import dopt.nnet;
import dopt.online;
import progress;

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
	import std.algorithm : joiner;
	import std.array : array;
	import std.format : format;
	import std.range : zip, chunks;
	import std.stdio : stderr, stdout, write, writeln;

	if(args.length != 2)
	{
		stderr.writeln("Usage: cifar10.d <data directory>");
		return;
	}

	/*
		Loads the CIFAR-10 dataset. Download this in the binary format from https://www.cs.toronto.edu/~kriz/cifar.html

		This also wraps the Dataset in an ImageTransformer, which will procedurally generate random crops and
		horizontal flips of the training images---a popular form of data augmentation for image datasets.
	*/
	writeln("Loading data...");
    auto data = new ImageTransformer(loadCIFAR10(args[1]), 4, 4, true, false);

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
	a classification task. The model is regularised using dropout, batch norm, and maxgain.
	*/
    auto preds = vgg19(features, [512, 512], true, true, 3.0f)
				.dense(10, new DenseOptions().maxgain(3.0f))
				.softmax();
    
	//The DAGNetwork class takes the inputs and outputs of a network and aggregates parameters in several different.
    auto network = new DAGNetwork([features], [preds]);

	/*
	Layer objects have both ``output'' and ``trainOutput'' fields, because operations like dropout perform different
	computations at train and test time. Therefore, we construct two different loss symbols: one for optimising, and
	one for evaluating.
	*/
    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLossSym = crossEntropy(preds.output, labels) + network.paramLoss;

	/*
	Now we set up an optimiser. Adam is good for proof of concepts, due to the fast convergence, however the
	performance of the final model is often slightly worse than that of a model trained with SGD+momentum.
	*/
	writeln("Creating optimiser...");
	auto learningRate = float32([], [0.0001f]);
	auto updater = adam([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate);
	auto testPlan = compile([testLossSym, preds.output]);

	writeln("Training...");

	float[] fs = new float[features.volume];
	float[] ls = new float[labels.volume];
	size_t bidx;

	//Iterate for 140 epochs of training!
	foreach(e; 0 .. 140)
	{
		float trainLoss = 0;
        float testLoss = 0;
        float trainAcc = 0;
        float testAcc = 0;
        float trainNum = 0;
        float testNum = 0;

		//Decreasing the learning rate after a while often results in better performance.
		if(e == 100)
		{
			learningRate.value.as!float[0] = 0.00001f;
		}
		else if(e == 120)
		{
			learningRate.value.as!float[0] = 0.000001f;
		}

		auto trainProgress = new Progress(data.foldSize(0) / batchSize);

		data.shuffle(0);

		do
		{
			//Get the next batch of training data (put into [fs, ls]). Update bidx with the next batch index.
			bidx = data.getBatch([fs, ls], bidx, 0);

			//Make an update to the model parameters using the minibatch of training data
			auto res = updater([
				features: Buffer(fs),
				labels: Buffer(ls)
			]);

			trainLoss += res[0].as!float[0] * batchSize;
			trainAcc += computeAccuracy(ls, res[1].as!float);
			trainNum += batchSize;

			float loss = trainLoss / trainNum;
			float acc = trainAcc / trainNum;

			trainProgress.title = format("Epoch: %03d  Loss: %02.4f  Acc: %.4f", e + 1, loss, acc);
            trainProgress.next();
		}
		while(bidx != 0);

		writeln();

		auto testProgress = new Progress(data.foldSize(1) / batchSize);

		do
		{
			//Get the next batch of testing data
			bidx = data.getBatch([fs, ls], bidx, 1);

			//Make some predictions
			auto res = testPlan.execute([
				features: Buffer(fs),
				labels: Buffer(ls)
			]);

			testLoss += res[0].as!float[0] * batchSize;
			testAcc += computeAccuracy(ls, res[1].as!float);
			testNum += batchSize;

			float loss = testLoss / testNum;
			float acc = testAcc / testNum;

			testProgress.title = format("            Loss: %02.4f  Acc: %.4f", loss, acc);
            testProgress.next();
		}
		while(bidx != 0);

		writeln();
		writeln();
	}
}

float computeAccuracy(float[] ls, float[] preds)
{
	import std.algorithm : maxIndex;
	import std.range : chunks, zip;

	float correct = 0;

	foreach(p, t; zip(preds.chunks(10), ls.chunks(10)))
	{
		if(p.maxIndex == t.maxIndex)
		{
			correct++;
		}
	}

	return correct;
}
