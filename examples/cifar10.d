#!/usr/bin/env dub
/+
dub.json:
{
    "name": "cifar10",
    "dependencies": {
        "dopt": {
            "path": "../"
        }
    }
}
+/
module cifar10;

void main(string[] args)
{
	import std.algorithm : joiner, maxIndex;
	import std.array : array;
	import std.range : zip, chunks;
	import std.stdio : stdout, write, writeln;

	import dopt.core;
	import dopt.nnet;
	import dopt.online;

    auto data = loadCIFAR10(args[1]);

    auto features = float32([100, 3, 32, 32]);
    auto labels = float32([100, 10]);

    auto preds = vgg16(features, [512, 512, 10]).softmax();
    
    auto network = new DAGNetwork([features], [preds]);

    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;
	auto testLoss = crossEntropy(preds.output, labels);

	auto learningRate = float32([], [0.0001f]);
	auto updater = adam([lossSym], network.params, network.paramProj, learningRate);

	foreach(e; 0 .. 120)
	{
		float tot = 0;

		if(e == 100)
		{
			learningRate.value.as!float[0] = 0.00001f;
		}

		foreach(fs, ls; zip(data.trainFeatures.chunks(100), data.trainLabels.chunks(100)))
		{
			updater([
				features: Buffer(fs.joiner().array()),
				labels: Buffer(ls.joiner().array())
			]);

			tot++;

            write("  ", tot, "/500    \r");
            stdout.flush();
		}

		int correct;
		int total;

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

		writeln(e + 1, ": ", correct / cast(float)total);
	}
}