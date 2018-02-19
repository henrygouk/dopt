module dopt.cuda.nnet;

import dopt.cuda.nnet.cudnn5;
import dopt.cuda.nnet.cudnn7;

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
