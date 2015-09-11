#include "Pool.h"

namespace ddnet
{
	//Constructor
	Pool::Pool(cudnnHandle_t handle, int wh, int ww, int vp, int hp, int vs, int hs, cudnnPoolingMode_t mode) : 
		Layer(handle), 
		mode_(mode), 
		wh_(wh), 
		ww_(ww), 
		vp_(vp), 
		hp_(hp), 
		vs_(vs), 
		hs_(hs) 
		{}

	//Set input Block
	void Pool::set_in(Block &block)
	{
		//Use a shared_ptr to keep track of the pooling descriptor
		cudnnPoolingDescriptor_t pptr;
		check( cudnnCreatePoolingDescriptor(&pptr) );
		check( cudnnSetPooling2dDescriptor(pptr,mode_,wh_,ww_,vp_,hp_,vs_,hs_) );
		descriptor_ = std::shared_ptr<cudnnPoolingStruct>(pptr,cudnnDestroyPoolingDescriptor);
		input = block;
	}

	//Set output Block
	void Pool::set_out(Block &block)
	{
		//NOTE: There is no checking function for pooling layer, this can break without exception!
		output = block;
	}

	Element& Pool::forward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Forward data
		float alpha = 1.0; float beta = 0.0;
		check( cudnnPoolingForward(handle,descriptor(),&alpha,
			input.data(i).descriptor(),input.data(i).raw(),
			&beta,
			output.data(i).descriptor(),output.data(i).raw()) );
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}
	//Backward the data
	Element& Pool::backward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Backward data
		float alpha = 1.0; float beta = 1.0;
		check( cudnnPoolingBackward(handle,descriptor(),&alpha,
			output.data(i).descriptor(),output.data(i).raw(),
			output.delta(i).descriptor(),output.delta(i).raw(),
			input.data(i).descriptor(),input.data(i).raw(),
			&beta,
			input.delta(i).descriptor(),input.delta(i).raw()) );
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}
}