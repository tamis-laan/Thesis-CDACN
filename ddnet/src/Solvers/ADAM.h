#pragma once
#include "./Solver.h"

namespace ddnet
{
	/*
		Adaptive Moment Estimation, improvement on RMSPROP
	*/
	class ADAM final : public Solver
	{
	public:
		//Constructor
		ADAM(float learning_rate = 0.00005, float decay_1 = 0.95, float decay_2 = 0.95, float epsilon = 1e-11)
		{
			meta_parameters_ = {learning_rate, decay_1, decay_2, epsilon};
		}
		//Return buffer values
		virtual std::vector<float> buffer_values() 
		{
			return {0.0,0.0};
		}
		//Apply ADAM
		virtual void step(Adapter &adaptor, cudaStream_t stream = NULL) override final;
	};
}