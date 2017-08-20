module dopt.core.grads.math;

import dopt.core.grads;
import dopt.core.ops;

package
{
    void initialize()
    {
        import std.functional;
        
        string createRegisterGradientCalls()
        {
            auto ops = ["add", "sub", "mul", "div", "pow", "min", "max",
                        "neg", "abs", "exp", "log", "sqrt"];

            string ret;

            foreach(o; ops)
            {
                ret ~= "registerGradient(\"" ~ o ~ "\", toDelegate(&" ~ o ~ "Grad));\n";
            }

            return ret;
        }

        mixin(createRegisterGradientCalls());

        registerGradient("matmul", toDelegate(&matmulGrad));
        registerGradient("sum", toDelegate(&sumGrad));
    }
}

private
{
    Operation[] matmulGrad(const(Operation) op, Operation parentGrad)
    {
        return [matmul(parentGrad, transpose(op.deps[1], [1, 0])), matmul(transpose(op.deps[0], [1, 0]), parentGrad)];
    }

    Operation[] sumGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad.repeat(op.deps[0].outputType.volume / op.outputType.volume)
                          .reshape(op.deps[0].outputType.shape)];
    }

    Operation[] addGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad, parentGrad];
    }

    Operation[] subGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad, neg(parentGrad)];
    }

    Operation[] mulGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad * op.deps[1], parentGrad * op.deps[0]];
    }

    Operation[] divGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            parentGrad / op.deps[1],
            neg(parentGrad * op.deps[0]) / (op.deps[1] * op.deps[1])
        ];
    }

    Operation[] powGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            parentGrad * op.deps[1] * pow(op.deps[0], op.deps[1] - 1),
            parentGrad * op.deps[1] * log(op.deps[0])
        ];
    }

    Operation[] minGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            op.deps[0].eq(op) * parentGrad,
            op.deps[1].eq(op) * parentGrad
        ];
    }

    Operation[] maxGrad(const(Operation) op, Operation parentGrad)
    {
        return [
            op.deps[0].eq(op) * parentGrad,
            op.deps[1].eq(op) * parentGrad
        ];
    }

    Operation[] negGrad(const(Operation) op, Operation parentGrad)
    {
        return [neg(parentGrad)];
    }

    Operation[] absGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad * sgn(op.deps[0])];
    }

    Operation[] expGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad * op];
    }

    Operation[] logGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad / op.deps[0]];
    }

    Operation[] sqrtGrad(const(Operation) op, Operation parentGrad)
    {
        return [parentGrad / op];
    }
}