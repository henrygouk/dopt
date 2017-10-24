module dopt.nnet.layers.datasource;

import dopt;

Layer dataSource(Operation var)
{
    return new Layer(null, var, var, null);
}