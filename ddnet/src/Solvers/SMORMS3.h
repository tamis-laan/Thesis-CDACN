#pragma once
#include "./Solver.h"

namespace ddnet
{
	/*
		Root Mean Square Back Propegation
	*/
	class SMORMS3 final : public Solver
	{
	public:
		//Constructor
		SMORMS3(float learning_rate = 0.1, float epsilon = 1e-16)
		{
			meta_parameters_ = {learning_rate, epsilon};
		}
		//Return buffer values
		virtual std::vector<float> buffer_values() 
		{ 
			return {1.0,0.0,0.0}; 
		}
		//Apply SMORMS3
		virtual void step(Adapter &adaptor, cudaStream_t stream = NULL) override final;
	};
}