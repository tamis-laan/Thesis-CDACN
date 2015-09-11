#include "Solver.h"

namespace ddnet
{
	//The step function applies a form of gradient descent.
	void Solver::step(Adapter &adaptor, cudaStream_t stream) 
	{
		for(int i=0; i<adaptor.size(); i++)
			cudnn_add(meta_parameters_[0],adaptor.gradients(i),1.0,adaptor.weights(i));
	}
}