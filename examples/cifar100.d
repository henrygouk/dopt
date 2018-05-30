#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
dependency "progress-d" version="~>1.0.0"
+/
module cifar100;

import dopt.core;
import dopt.nnet;
import dopt.online;
import progress;

void main(string[] args)
{
	import std.algorithm : joiner;
	import std.array : array;
	import std.format : format;
	import std.range : zip, chunks;
	import std.stdio : stderr, stdout, write, writeln;

	if(args.length != 2)
	{
		stderr.writeln("Usage: cifar100.d <data directory>");
		return;
	}

    writeln("Loading data...");
    auto data = loadCIFAR100(args[1]);
	data.train = new ImageTransformer(data.train, 4, 4, true, false);

    writeln("Constructing network graph...");
    size_t batchSize = 48;
    auto features = float32([batchSize, 3, 32, 32]);
    auto labels = float32([batchSize, 100]);

    auto preds = wideResNet(features, 16, 4)
                .dense(100)
                .softmax();

    auto network = new DAGNetwork([features], [preds]);
    
    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLossSym = crossEntropy(preds.output, labels) + network.paramLoss;

    writeln("Creating optimiser...");
	auto learningRate = float32(0.1f);
	auto momentumRate = float32(0.9);
	auto updater = sgd([lossSym, preds.trainOutput], network.params, network.paramProj, learningRate, momentumRate);
	auto testPlan = compile([testLossSym, preds.output]);

	writeln("Training...");

	float[] fs = new float[features.volume];
	float[] ls = new float[labels.volume];
	size_t bidx;

	//Iterate for 160 epochs of training!
	foreach(e; 0 .. 200)
	{
		float trainLoss = 0;
        float testLoss = 0;
        float trainAcc = 0;
        float testAcc = 0;
        float trainNum = 0;
        float testNum = 0;

		//Decreasing the learning rate after a while often results in better performance.
		if(e == 60)
		{
			learningRate.value.set([0.02f]);
		}
		else if(e == 120)
		{
			learningRate.value.set([0.004f]);
		}
        else if(e == 160)
        {
            learningRate.value.set([0.0008f]);
        }

		data.train.restart();
		data.test.restart();

		auto trainProgress = new Progress(data.train.length / batchSize);

		while(!data.train.finished())
		{
			//Get the next batch of training data (put into [fs, ls]). Update bidx with the next batch index.
			data.train.getBatch([fs, ls]);

			//Make an update to the model parameters using the minibatch of training data
			auto res = updater([
				features: buffer(fs),
				labels: buffer(ls)
			]);

			trainLoss += res[0].get!float[0] * batchSize;
			trainAcc += computeAccuracy(ls, res[1].get!float);
			trainNum += batchSize;

			float loss = trainLoss / trainNum;
			float acc = trainAcc / trainNum;

			trainProgress.title = format("Epoch: %03d  Loss: %02.4f  Acc: %.4f", e + 1, loss, acc);
            trainProgress.next();
		}

		writeln();

		auto testProgress = new Progress(data.test.length / batchSize);

		while(!data.test.finished)
		{
			//Get the next batch of testing data
			data.test.getBatch([fs, ls]);

			//Make some predictions
			auto res = testPlan.execute([
				features: buffer(fs),
				labels: buffer(ls)
			]);

			testLoss += res[0].get!float[0] * batchSize;
			testAcc += computeAccuracy(ls, res[1].get!float);
			testNum += batchSize;

			float loss = testLoss / testNum;
			float acc = testAcc / testNum;

			testProgress.title = format("            Loss: %02.4f  Acc: %.4f", loss, acc);
            testProgress.next();
		}

		writeln();
		writeln();
	}
}

float computeAccuracy(float[] ls, float[] preds)
{
	import std.algorithm : maxIndex;
	import std.range : chunks, zip;

	float correct = 0;

	foreach(p, t; zip(preds.chunks(100), ls.chunks(100)))
	{
		if(p.maxIndex == t.maxIndex)
		{
			correct++;
		}
	}

	return correct;
}
