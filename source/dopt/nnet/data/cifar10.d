module dopt.nnet.data.cifar10;

import std.algorithm;
import std.array;
import std.file;
import std.range;
import std.stdio;
import std.typecons;

import dopt;

Dataset loadCIFAR10(string path)
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

	return Dataset(features[0 .. 50_000], features[50_000 .. $], labels[0 .. 50_000], labels[50_000 .. $]);
}