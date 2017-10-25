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

import std.algorithm;
import std.array;
import std.file;
import std.range;
import std.stdio;
import std.typecons;

auto loadCIFAR10(string path)
{
	auto batches = ["data_batch_1.bin",
		 			"data_batch_2.bin",
					"data_batch_3.bin",
					"data_batch_4.bin",
					"data_batch_5.bin",
					"test_batch.bin"].map!(x => path ~ "/" ~ x).array();

	alias T = float;
	T[][] features;
	T[][] labels;

	foreach(b; batches)
	{
		ubyte[] raw = cast(ubyte[])read(b);

		for(size_t i = 0; i < 10_000; i++)
		{
			auto f = raw[1 .. 32 * 32 * 3 + 1]
				.map!(x => (cast(T)x - 128.0f) / 48.0f)
				.array();

			auto ls = new T[10];
			ls[] = 0;
			ls[raw[0]] = 1.0f;
			labels ~= ls;
			features ~= f;

			raw = raw[32 * 32 * 3 + 1 .. $];
		}
	}

	return tuple!("trainFeatures", "trainLabels", "testFeatures", "testLabels")
				 (features[0 .. 50_000], labels[0 .. 50_000], features[50_000 .. $], labels[50_000 .. $]);
}
	
import dopt.core;
import dopt.nnet;
import dopt.online;

Layer vggBlock(Layer input, size_t channels)
{
    return input
          .conv2D(channels, [3, 3], new Conv2DOptions().padding([1, 1]))
          .relu()
          .conv2D(channels, [3, 3], new Conv2DOptions().padding([1, 1]))
          .relu()
          .maxPool([2, 2]);
}

void main(string[] args)
{
    auto data = loadCIFAR10(args[1]);

    auto features = float32([100, 3, 32, 32]);
    auto labels = float32([100, 10]);

    auto preds = dataSource(features)
                .vggBlock(64)
                .vggBlock(128)
                .vggBlock(256)
                .vggBlock(512)
                .vggBlock(512)
                .dense(512)
                .relu()
                .dense(512)
                .relu()
                .dense(10)
                .softmax();
    
    auto network = new DAGNetwork([features], [preds]);

    auto lossSym = crossEntropy(preds.trainOutput, labels) + network.paramLoss;

	auto learningRate = float32([], [0.0001f]);
	auto updater = adam([lossSym], network.params, null, learningRate);

	foreach(e; 0 .. 120)
	{
		float totloss = 0;
		float tot = 0;

		if(e == 100)
		{
			learningRate.value.as!float[0] = 0.00001f;
		}

		foreach(fs, ls; zip(data.trainFeatures.chunks(100), data.trainLabels.chunks(100)))
		{
			auto loss = updater([
				features: Buffer(fs.joiner().array()),
				labels: Buffer(ls.joiner().array())
			]);

			totloss += loss[0].as!float[0];
			tot++;

            write("  ", tot, "/500    \r");
            stdout.flush();
		}

		writeln();
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