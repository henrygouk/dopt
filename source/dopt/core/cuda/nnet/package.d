module dopt.core.cuda.nnet;

import dopt.core.cuda.nnet.cudnn5;
import dopt.core.cuda.nnet.cudnn7;

void initialize()
{
    try
	{
		initializeCuDNN7();
	}
	catch(Exception e)
	{
		try
		{
			initializeCuDNN5();
		}
		catch(Exception e)
		{
		}
	}
}
