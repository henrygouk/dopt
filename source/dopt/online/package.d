/**
    This package contains implementations of common online optimisation algorithms, with a particular bias towards
    those commonly used in machine learning/deep learning.

    Authors: Henry Gouk
*/
module dopt.online;

public
{
    import dopt.online.adam;
    import dopt.online.sgd;
}

import dopt.core;

/**
    Used for performing projected gradient descent.

    The delegate should take a new value for some tensor, and project it back into the feasible set.
*/
alias Projection = Operation delegate(Operation);

/**
    A delegate that can be used to perform the update step for an online optimisation algorithm.
*/
alias Updater = Buffer[] delegate(Buffer[Operation] args);