#pragma once
#include "./Solver.h"

namespace ddnet
{
	/*
		Batch Gradient Descent
	*/
	class BGD final : public Solver
	{
	public:
		//Constructor
		BGD(float learning_rate = 0.007, float momentum = 0.7)
		{
			meta_parameters_ = {learning_rate, momentum};
		}
		//Return buffer values
		virtual std::vector<float> buffer_values() 
		{
			return {0.0};
		}
		//Apply Batch Gradient Descent with Momentum
		virtual void step(Adapter &adaptor, cudaStream_t stream = NULL) override final;
	};
}