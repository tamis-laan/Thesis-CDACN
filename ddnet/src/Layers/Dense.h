#pragma once
#include "Convolution.h"

namespace ddnet
{
	class Dense final : public Convolution
	{
	public:
		//Default Constructor
		Dense() = default;
		//Constructor
		Dense(cudnnHandle_t handle, int neurons, float f_var = 0.01, float b_val = 0.02);
		//Set input Block
		virtual void set_in(Block &block) override;
		//Set output Block
		virtual void set_out(Block &block) override;
		//Apply forward computation
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Compute the data derivative
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
		//Compute the filter derivative
		virtual Element& update(int i, bool accumulate = false, cudaStream_t stream = NULL) override;
		//Return filter and bias
		virtual std::vector<Tensor> weights() override;
		//Return filter and bias gradients
		virtual std::vector<Tensor> gradients() override;
	};
}