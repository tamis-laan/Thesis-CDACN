#pragma once
#include "Layer.h"

namespace ddnet
{
	class TestLayer : public Layer
	{
	private:
		TensorArray delta_avrg;
	public:
		TestLayer(cudnnHandle_t handle) : Layer(handle) {} 

		//Set input Block
		virtual void set_in(Block &block) override
		{
			//Set input block
			input = block;
			//Construct same size block
			delta_avrg = TensorArray(handle, block.n(),block.c(),block.h(),block.w(),1.0);
		}
		//Set output Block
		virtual void set_out(Block &block) override
		{
			//Check same batch size
			if(input.n()!=block.n()) throw ddnet_exception(0);
			//Check equal dimentions
			if(input.c()*input.h()*input.w() != block.c()*block.h()*block.w()) throw ddnet_exception(0);
			//Set output block
			output = block;
		}
		//Forward data
		virtual Element& forward(int i, cudaStream_t stream = NULL) override
		{
			//just input to output
			copy(input.data(i),output.data(i));

			return *this;
		}
		//Backward data
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
	};	
}