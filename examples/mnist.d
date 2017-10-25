#!/usr/bin/env dub
/+
dub.json:
{
    "name": "mnist",
    "dependencies": {
        "dopt": {
            "path": "../"
        }
    }
}
+/
module mnist;

import std.algorithm;
import std.array;
import std.file;
import std.range;
import std.typecons;

auto loadMNIST(string path)
{
	auto trainFeatures = loadFeatures!float(path ~ "/train-images-idx3-ubyte");
	auto trainLabels = loadLabels!float(path ~ "/train-labels-idx1-ubyte");
	auto testFeatures = loadFeatures!float(path ~ "/t10k-images-idx3-ubyte");
	auto testLabels = loadLabels!float(path ~ "/t10k-labels-idx1-ubyte");

	return tuple!("trainFeatures", "testFeatures", "trainLabels", "testLabels")
					(trainFeatures, testFeatures, trainLabels, testLabels);
}

T[][] loadFeatures(T)(string filename)
{
	const size_t numFeatures = 28 * 28;

	//Load the data from disk
	ubyte[] raw = cast(ubyte[])read(filename);

	//Skip over the header
	raw = raw[16 .. $];

	//Get the number of instances in this file
	size_t numInstances = raw.length / numFeatures;

	//Allocate space to store the references to each instance
	T[][] result = new T[][numInstances];

	//Convert the ubytes to floats
	T[] features = raw.map!(x => cast(T)x / cast(T)255.0).array();

	//Iterate over each instance and set the references to the correct slice
	for(size_t i = 0; i < numInstances; i++)
	{
		result[i] = features[i * numFeatures .. (i + 1) * numFeatures];
	}

	return result;
}

T[][] loadLabels(T)(string filename)
{
	const size_t numLabels = 10;

	//Load the data from disk
	ubyte[] raw = cast(ubyte[])read(filename);

	//Skip over the header
	raw = raw[8 .. $];

	//Get the number of instances in this file
	size_t numInstances = raw.length;

	//Allocate space to store the references to each instance
	T[][] result = new T[][numInstances];
	T[] labels = new T[numInstances * numLabels];
	labels[] = 0.0;

	//Create the one-hot encoding array and set up references to the appropriate slices for each instance
	for(size_t i = 0; i < numInstances; i++)
	{
		result[i] = labels[i * numLabels .. (i + 1) * numLabels];
		result[i][raw[i]] = 1.0;
	}

	return result;
}

void main(string[] args)
{
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