module dopt.nnet.data.sins;

import std.algorithm : map;
import std.array : array;
import std.file : read;
import std.range : chunks, zip;
import std.typecons : tuple;

import dopt.nnet.data;

Dataset[] loadSINS10(string directory)
{
    auto features = (cast(ubyte[])read(directory ~ "/X.bin"))
                   .map!(x => cast(float)x / 128.0f - 1.0f)
                   .chunks(96 * 96 * 3)
                   .map!(x => x.array)
                   .array();
    
    auto rawlabels = (cast(ubyte[])read(directory ~ "/y.bin"));

    float[][] labels = new float[][rawlabels.length];

    for(size_t i = 0; i < labels.length; i++)
    {
        labels[i] = new float[10];
        labels[i][] = 0;
        labels[i][rawlabels[i]] = 1.0f;
    }

    return zip(features.chunks(10_000), labels.chunks(10_000))
          .map!(x => tuple(x[0].array, x[1].array))
          .map!(x => new CORDataset(
              [x[0][0 .. 9_000], x[0][9_000 .. $]],
              [x[1][0 .. 9_000], x[1][9_000 .. $]],
              [3, 96, 96]))
          .map!(x => cast(Dataset)x)
          .array();
}