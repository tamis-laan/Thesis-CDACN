#include "Activation.h"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#include <iostream>
using namespace std;

namespace ddnet
{

	///////////////////////////////////////
	///////////// Leaky ReLU //////////////
	///////////////////////////////////////

	struct lrelu_forward
	{
		float  l_;
		float* input_;
		float* output_;
		lrelu_forward(float* input, float* output, float l) : input_(input), output_(output), l_(l){}
		__device__ void operator()(const int i)
		{
			float max = (input_[i] >  0)*input_[i];
			float min = (input_[i] <= 0)*l_*input_[i];
			output_[i] = max + min;
		}
		
	};

	struct lrelu_backward
	{
		float  l_;
		float* input_;
		float* input_grad_;
		float* output_;
		float* output_grad_;
		lrelu_backward(float* input, float* input_grad, float* output, float* output_grad, float l) : input_(input), input_grad_(input_grad), output_(output), output_grad_(output_grad), l_(l) {}
		__device__ void operator()(const int i)
		{
			float max  = (input_[i] >  0) * output_grad_[i];
			float min  = (input_[i] <= 0) * l_ * output_grad_[i];
			input_grad_[i] = max + min;
		}
	};

	///////////////////////////////////////
	/////////// Parametric ReLU ///////////
	///////////////////////////////////////
	struct prelu_forward
	{
		float* buffer_;
		float* input_;
		float* output_;
		int n_;
		prelu_forward(float* input, float* output, float* buffer, int n) : input_(input), output_(output), buffer_(buffer), n_(n){}
		__device__ void operator()(const int i)
		{
			float max = (input_[i] >  0)*input_[i];
			float min = (input_[i] <= 0)*buffer_[i%n_]*input_[i];
			output_[i] = max + min;
		}
	};

	struct prelu_backward
	{
		float* input_;
		float* input_grad_;
		float* output_;
		float* output_grad_;
		float* buffer_;
		int n_;
		prelu_backward(float* input, float* input_grad, float* output, float* output_grad, float* buffer, int n) : input_(input), input_grad_(input_grad), output_(output), output_grad_(output_grad), buffer_(buffer), n_(n) {}
		__device__ void operator()(const int i)
		{
			float max  = (input_[i] >  0)*output_grad_[i];
			float min  = (input_[i] <= 0)*buffer_[i%n_]*output_grad_[i];
			input_grad_[i] = max + min;
		}
	};

	struct prelu_update
	{
		float* input_;			//Input
		float* output_grad_;	//Output gradient
		prelu_update(float* input, float* output_grad) : input_(input), output_grad_(output_grad) {}
		__device__ float operator()(const int i)
		{
			return output_grad_[i]*(input_[i]<=0)*input_[i];
		}
	};

	// struct prelu_update
	// {
	// 	float* input_;			//Input
	// 	float* output_grad_;	//Output gradient
	// 	float* buffer_grad_;	//Buffer gradient
	// 	int n_;					//Input size
	// 	prelu_update(float* input, float* output_grad, float* buffer_grad, int n) : input_(input), output_grad_(output_grad), buffer_grad_(buffer_grad), n_(n) {}
	// 	__device__ void operator()(const int i)
	// 	{
	// 		float sum = 0;
	// 		for(int j=0; j<n_; j++)
	// 			sum+=output_grad_[j]*(input_[j]<=0)*input_[j];
	// 		buffer_grad_[i] = sum;
	// 	}
	// };

	//Default Constructor
	Activation::Activation() : mode_(Mode::ReLU) 
	{}
	
	//Consructor
	Activation::Activation(cudnnHandle_t handle, Mode mode) : Layer(handle), mode_(mode) 
	{}

	//Set input Block
	void Activation::set_in(Block &block)
	{
		input = block;
		if(mode_==Mode::PReLU)
		{
			int size = input.data().c()*input.data().h()*input.data().w();
			buffer_       = Tensor(handle,1,size,0.25);
			buffer_grad_  = Tensor(handle,1,size,0.25);
		}
	}

	//Set output Block
	void Activation::set_out(Block &block)
	{
		if(input.n()!=block.n()) throw ddnet_exception(0);
		if(input.c()*input.h()*input.w() != block.c()*block.h()*block.w()) throw ddnet_exception(0);
		output = block;
	}

	//Forward data
	Element& Activation::forward(int i, cudaStream_t stream) 
	{
		//CUDA CHECK
		if(mode_==Mode::ReLU or mode_==Mode::sigmoid or mode_==Mode::tanh)
		{
			//Set the stream
			check( cudnnSetStream(handle,stream) );
			//Forward data
			float alpha = 1.0; float beta = 0.0;
			if(mode_==Mode::sigmoid)
				check( cudnnActivationForward(handle,CUDNN_ACTIVATION_SIGMOID,&alpha,input.data(i).descriptor(),input.data(i).raw(),&beta,output.data(i).descriptor(),output.data(i).raw()) );
			if(mode_==Mode::ReLU)
				check( cudnnActivationForward(handle,CUDNN_ACTIVATION_RELU,&alpha,input.data(i).descriptor(),input.data(i).raw(),&beta,output.data(i).descriptor(),output.data(i).raw()) );
			if(mode_==Mode::tanh)
				check( cudnnActivationForward(handle,CUDNN_ACTIVATION_TANH,&alpha,input.data(i).descriptor(),input.data(i).raw(),&beta,output.data(i).descriptor(),output.data(i).raw()) );
			//Set back to NULL stream
			check( cudnnSetStream(handle,NULL) );
		}
		if(mode_==Mode::PReLU)
		{
			thrust::counting_iterator<int> first(0);
			thrust::counting_iterator<int> last(input.size(i));
			thrust::for_each(first, last, prelu_forward(input.data(i).raw(),output.data(i).raw(),buffer_.raw(),buffer_.size()) );
		}
		if(mode_==Mode::LReLU)
		{
			thrust::counting_iterator<int> first(0);
			thrust::counting_iterator<int> last(input.size(i));
			thrust::for_each(first, last, lrelu_forward(input.data(i).raw(),output.data(i).raw(),0.25f));	
		}
		return *this;
	}

	//Backward data
	Element& Activation::backward(int i, cudaStream_t stream) 
	{
		//CUDA CHECK
		if(mode_==Mode::ReLU or mode_==Mode::sigmoid or mode_==Mode::tanh)
		{
			//Set the stream
			check( cudnnSetStream(handle,stream) );
			//Backward data
			float alpha = 1.0; float beta = 0.0;
			if(mode_==Mode::sigmoid)
				check( cudnnActivationBackward(handle,CUDNN_ACTIVATION_SIGMOID,&alpha,output.data(i).descriptor(),output.data(i).raw(),output.delta(i).descriptor(),output.delta(i).raw(),input.data(i).descriptor(),input.data(i).raw(),&beta,input.delta(i).descriptor(),input.delta(i).raw()) );
			if(mode_==Mode::ReLU)
				check( cudnnActivationBackward(handle,CUDNN_ACTIVATION_RELU,&alpha,output.data(i).descriptor(),output.data(i).raw(),output.delta(i).descriptor(),output.delta(i).raw(),input.data(i).descriptor(),input.data(i).raw(),&beta,input.delta(i).descriptor(),input.delta(i).raw()) );
			if(mode_==Mode::tanh)
				check( cudnnActivationBackward(handle,CUDNN_ACTIVATION_TANH,&alpha,output.data(i).descriptor(),output.data(i).raw(),output.delta(i).descriptor(),output.delta(i).raw(),input.data(i).descriptor(),input.data(i).raw(),&beta,input.delta(i).descriptor(),input.delta(i).raw()) );
			//Set back to NULL stream
			check( cudnnSetStream(handle,NULL) );
		}
		if(mode_==Mode::PReLU)
		{
			thrust::counting_iterator<int> first(0);
			thrust::counting_iterator<int> last(input.size(i));
			thrust::for_each(first, last, prelu_backward(input.data(i).raw(),input.delta(i).raw(),output.data(i).raw(),output.delta(i).raw(),buffer_.raw(),buffer_.size()) );
		}
		if(mode_==Mode::LReLU)
		{
			thrust::counting_iterator<int> first(0);
			thrust::counting_iterator<int> last(input.size(i));
			thrust::for_each(first, last, lrelu_backward(input.data(i).raw(),input.delta(i).raw(),output.data(i).raw(),output.delta(i).raw(),0.25f) );
		}
		return *this;
	}

	//Update gradient
	Element& Activation::update(int i, bool accumulate, cudaStream_t stream) 
	{
		if(mode_==Mode::PReLU)
		{
			thrust::counting_iterator<int> first(0);
			thrust::counting_iterator<int> last(buffer_.size());
			float sum = thrust::transform_reduce(first, last, prelu_update(input.data(i).raw(),output.delta(i).raw()), 0.0f, thrust::plus<float>());
			cudnn_fill(sum,buffer_grad_);
		}
		return *this;
	}

	//Return filter and bias
	std::vector<Tensor> Activation::weights()
	{
		if(mode_==Mode::PReLU)
			return {buffer_};
		else 
			return {};
	}

	//Return filter and bias gradients
	std::vector<Tensor> Activation::gradients() 
	{
		if(mode_==Mode::PReLU)
			return {buffer_grad_};
		else
			return {};
	}	


}