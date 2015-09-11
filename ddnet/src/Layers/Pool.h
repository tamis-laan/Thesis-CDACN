#pragma once
#include "Layer.h"

namespace ddnet
{
	class Pool final : public Layer
	{
	protected:
		/* Modes are {CUDNN_POOLING_MAX , CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING , CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING} */
		cudnnPoolingMode_t mode_;
		std::shared_ptr<cudnnPoolingStruct> descriptor_;
		int wh_; int ww_; int vp_; int hp_; int vs_; int hs_;
		//Return the pooling descriptor
		cudnnPoolingDescriptor_t descriptor() {return descriptor_.get();}
	public:
		//Default Constructor
		Pool() = default;
		//Constructor
		Pool(cudnnHandle_t handle, int wh=2, int ww=2, int vp=0, int hp=0, int vs=1, int hs=1, cudnnPoolingMode_t mode=CUDNN_POOLING_MAX);
		//Set input Block
		virtual void set_in(Block &block) override;
		//Set output Block
		virtual void set_out(Block &block) override;
		//Forward the data
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Backward the data
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
	};
}