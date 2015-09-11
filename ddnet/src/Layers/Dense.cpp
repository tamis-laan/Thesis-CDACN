#include "Dense.h"

namespace ddnet
{
	//Constructor
	Dense::Dense(cudnnHandle_t handle, int neurons, float f_var, float b_val) : Convolution(handle,neurons,1,1,1,1,f_var,b_val,0,0) 
	{}

	//Set input Block
	void Dense::set_in(Block &block)
	{
		//Set the input block
		input = block;
		//Reshape
		input.reshape(input.n(),1,1,input.c()*input.h()*input.w());
		//Set filter width correctly
		fw_ = input.w();
		//Call base
		Convolution::set_in(input);
	}

	//Set output Block
	void Dense::set_out(Block &block)
	{
		//Set output block
		output = block;
		//Reshape
		output.reshape(output.n(),output.c()*output.h()*output.w(),1,1);
		//Call base
		Convolution::set_out(output);
	}

	//Apply forward computation
	Element& Dense::forward(int i, cudaStream_t stream)
	{
		return Convolution::forward(i, stream);
	}

	//Compute the data derivative
	Element& Dense::backward(int i, cudaStream_t stream)
	{
		return Convolution::backward(i, stream);
	}

	//Compute the filter derivative
	Element& Dense::update(int i, bool accumulate, cudaStream_t stream)
	{
		return Convolution::update(i, accumulate, stream);
	}

	//Return filter and bias
	std::vector<Tensor> Dense::weights()
	{
		return Convolution::weights();
	}

	//Return filter and bias gradients
	std::vector<Tensor> Dense::gradients()
	{
		return Convolution::gradients();
	}
}