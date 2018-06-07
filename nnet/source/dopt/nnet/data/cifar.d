module dopt.nnet.data.cifar;

import std.algorithm;
import std.array;
import std.file;
import std.range;
import std.stdio;
import std.typecons;

import dopt.nnet.data;

auto loadCIFAR10(string path, bool validation = false)
{
	auto batchFiles = [
		"data_batch_1.bin",
		"data_batch_2.bin",
		"data_batch_3.bin",
		"data_batch_4.bin",
		"data_batch_5.bin",
		"test_batch.bin"
	];

	return loadCIFAR(path, batchFiles, 1, 0, 10, validation);
}

auto loadCIFAR100(string path, bool validation = false)
{
	auto batchFiles = ["train.bin", "test.bin"];

	return loadCIFAR(path, batchFiles, 2, 1, 100, validation);
}

private
{
	auto loadCIFAR(string path, string[] batchFiles, size_t labelBytes, size_t labelIdx, size_t numLabels, bool valid)
	{
		auto batches = batchFiles.map!(x => path ~ "/" ~ x).array();

		alias T = float;
		T[][] features;
		T[][] labels;

		foreach(b; batches)
		{
			ubyte[] raw = cast(ubyte[])read(b);

			foreach(tmp; raw.chunks(3 * 32 * 32 + labelBytes))
			{
				auto f = tmp[labelBytes .. $]
					.map!(x => cast(T)x / 128.0f - 1.0f)
					.array();

				auto ls = new T[numLabels];
				ls[] = 0;
				ls[tmp[labelIdx]] = 1.0f;
				labels ~= ls;
				features ~= f;
			}
		}

		size_t numTrain = valid ? 40_000 : 50_000;

		BatchIterator trainData = new SupervisedBatchIterator(
			features[0 .. numTrain],
			labels[0 .. numTrain],
			[[cast(size_t)3, 32, 32], [numLabels]],
			true
		);

		BatchIterator testData = new SupervisedBatchIterator(
			features[numTrain .. numTrain + 10_000],
			labels[numTrain .. numTrain + 10_000],
			[[cast(size_t)3, 32, 32], [numLabels]],
			false
		);

		import std.typecons;
    
    	return tuple!("train", "test")(trainData, testData);
	}
}