module dopt.core.types;

import std.algorithm;

enum DataType
{
    float32,
    int32
}

size_t sizeOf(DataType t)
{
    switch(t)
    {
        case DataType.float32:
            return float.sizeof;

        case DataType.int32:
            return int.sizeof;

        default:
            throw new Exception("Not implemented");
    }
}

struct TensorType
{
    DataType elementType;
    size_t[] shape;

    this(DataType et, const(size_t)[] s)
    {
        elementType = et;
        shape = s.dup;
    }

    this(const TensorType t)
    {
        elementType = t.elementType;
        shape = t.shape.dup;
    }

    bool opEquals(const TensorType t) const
    {
        return elementType == t.elementType && shape == t.shape;
    }

    @property size_t rank() const
    {
        return shape.length;
    }

    @property size_t volume() const
    {
        return shape.fold!((a, b) => a * b)(cast(size_t)1);
    }
}

/**
Creates a wrapper around a user allocated buffer
*/
struct Buffer
{
    public
    {
        this(void[] buf)
        {
            mBuffer = buf;
        }

        T[] as(T)()
        {
            return cast(T[])mBuffer;
        }

        const(T[]) as(T)() const
        {
            return cast(const(T[]))mBuffer;
        }
    }

    private
    {
        void[] mBuffer;
    }
}