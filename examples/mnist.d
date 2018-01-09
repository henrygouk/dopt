#!/usr/bin/env dub
/+ dub.sdl:
dependency "dopt" path=".."
+/
module mnist;

void main(string[] args)
{
	import std.algorithm : joiner, maxIndex;
	import std.array : array;
	import std.range : zip, chunks;
	import std.stdio : writeln;
	
	import dopt.core;
	import dopt.nnet;
	import dopt.online;

    auto data = loadMNIST(args[1]);

    auto features = float32([100, 1, 28, 28]);
	auto labels = float32([100, 10]);

	auto preds = dataSource(features)
				.conv2D(32, [5, 5])
				.relu()
				.maxPool([2, 2])
				.conv2D(32, [5, 5])
				.relu()
				.maxPool([2, 2])
				.dense(10)
				.softmax();

	auto network = new DAGNetwork([features], [preds]);

	auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;

	auto updater = adam([lossSym], network.params, null);

	foreach(e; 0 .. 10)
	{
		float totloss = 0;
		float tot = 0;

		foreach(fs, ls; zip(data.trainFeatures.chunks(100), data.trainLabels.chunks(100)))
		{
			auto loss = updater([
				features: Buffer(fs.joiner().array()),
				labels: Buffer(ls.joiner().array())
			]);

			totloss += loss[0].as!float[0];
			tot++;
		}

		writeln(e, ": ", totloss / tot);
	}

	int correct;
	int total;

	import std.stdio : writeln;

	foreach(fs, ls; zip(data.testFeatures.chunks(100), data.testLabels.chunks(100)))
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

	writeln(correct / cast(float)total);
}