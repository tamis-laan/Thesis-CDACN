#pragma once
#include "./Solver.h"

namespace ddnet
{
	/*
		Root Mean Square Back Propegation
	*/
	class RMSPROP final : public Solver
	{
	public:
		//Constructor
		RMSPROP(float learning_rate = 0.00005, float decay_1 = 0.95, float decay_2 = 0.95, float epsilon = 1e-08)
		{
			meta_parameters_ = {learning_rate, decay_1, decay_2, epsilon};
		}
		//Return buffer values
		virtual std::vector<float> buffer_values() 
		{ 
			return {0.0,0.0}; 
		}
		//Apply RMSPROP
		virtual void step(Adapter &adaptor, cudaStream_t stream = NULL) override final;
	};
}