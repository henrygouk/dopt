module dopt.nnet.util;

string dynamicProperties(Args...)(Args args)
{
    static if(args.length > 0)
    {
        return "
public " ~ args[0] ~ " " ~ args[1] ~ "() { return _" ~ args[1] ~ "; }
public typeof(this) " ~ args[1] ~ "(" ~ args[0] ~ " p) { _" ~ args[1] ~ " = p; return this; }
private " ~ args[0] ~ " _" ~ args[1] ~ ";
        " ~ dynamicProperties(args[2 .. $]);
    }
    else
    {
        return "";
    }
}