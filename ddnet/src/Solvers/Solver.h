#pragma once
#include "../Layers/Layer.h"
#include "../Adapter.h"

namespace ddnet
{
	/*
		The solver class is a base class for any gradient descent method that applies itselft to a Adapter/Element in the step function.
		The basic base class implement Batch Gradient Descent without momentum. 
	*/
	class Solver
	{
	protected:
		//Meta-parameters that can be set
		std::vector<float> meta_parameters_;
	public:
		//Constructor takes in default parameters
		Solver(float learning_rate = 0.01)
		{
			meta_parameters_ = {learning_rate};
		}
		//Get/Set meta parameter
		float& meta_parameter(int i) 
		{
			return meta_parameters_[i];
		}
		//Return buffer values
		virtual std::vector<float> buffer_values()
		{
			return {};
		}
		//The step function applies a form of gradient descent.
		virtual void step(Adapter &adaptor, cudaStream_t stream = NULL); //<--- ADD STREAMING!
	};
}