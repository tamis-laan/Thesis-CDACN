#pragma once
#include "Layer.h"

namespace ddnet
{
	class Activation : public Layer
	{
	public:
		//Activation enum
		enum class Mode{ReLU,PReLU,LReLU,sigmoid,tanh};
		//Default Constructor
		Activation();
		//Constructor
		Activation(cudnnHandle_t handle, Mode mode = Mode::ReLU);
		//Set input Block
		virtual void set_in(Block &block) override;
		//Set output Block
		virtual void set_out(Block &block) override;
		//Forward data
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Backward data
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
		//Update parameters
		virtual Element& update(int i, bool accumulate = false, cudaStream_t stream = NULL);
		//Return buffer
		virtual std::vector<Tensor> weights() override;
		//Return buffer gradients
		virtual std::vector<Tensor> gradients() override;
	protected:
		//Activation mode used
		Mode mode_;
		//Buffers 
		Tensor buffer_;
		//Buffer gradient
		Tensor buffer_grad_;
	};
}