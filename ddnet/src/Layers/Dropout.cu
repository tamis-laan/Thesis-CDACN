#include "Dropout.h"
#include <thrust/random.h>
#include <thrust/transform.h>
#include <thrust/device_vector.h>

namespace ddnet
{
	Dropout::Dropout(cudnnHandle_t handle, float prob, std::vector<bool> p) : Layer(handle), prob_(prob), p_(p) 
	{
		if(prob_<0.0 or prob_>1.0)
			throw ddnet_exception(1);
		srand(time(0));
	}

	void Dropout::set_in(Block &block)
	{
		if(p_.size() == 0)
		{
			p_ = std::vector<bool>(block.n().size(),false);
			for(int i=0; i<p_.size(); i++)
				p_[i] = (block.n()[i]==1) ? true : false;
		}
		if(block.n().size()!=p_.size()) throw ddnet_exception(0);
		input = block;
	}

	//Set output Block
	void Dropout::set_out(Block &block)
	{	
		if(input.n()!=block.n()) throw ddnet_exception(0);
		if(input.c()*input.h()*input.w() != block.c()*block.h()*block.w()) throw ddnet_exception(0);
		output = block;
		binary_ = TensorArray(handle,block.n(),block.c(),block.h(),block.w());
	}

	//Generate random bit string
	struct random_kernel
	{
		float seed;
		float threshold;
		random_kernel(float seed, float threshold) : seed(seed), threshold(threshold) {}
		__device__ float operator()(int idx)
		{
			thrust::random::ranlux24_base engine(seed);
			thrust::uniform_real_distribution<float> dist;
			engine.discard(idx);
			return dist(engine)<threshold;	
		}
	};

	//Apply dropout or in the case of prediction multiply by dropout probability threshold
	struct forward_kernel
	{
		float threshold;
		bool predict;
		forward_kernel(float threshold, bool predict) : threshold(threshold), predict(predict) {}
		__device__ float operator()(float &x, float &y)
		{
			//Predict or Dropout
			return predict ? y*threshold : x*y;
		}
	};

	Element& Dropout::forward(int i, cudaStream_t stream)
	{
		/*NOTE: Use streamed versions when cuda 7.0 becomes available!*/
		// thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(binary_(i).size()), binary_(i).begin(), random_kernel(rand(),prob_));
		thrust::transform(binary_(i).begin(), binary_(i).end(), input.data(i).begin(), output.data(i).begin(), forward_kernel(prob_,p_[i]));
		return *this;
	}

	//Element multiply binary string with delta
	struct backward_kernel
	{
		__device__ float operator()(float &x, float &y)
		{   return x*y;   }
	};

	Element& Dropout::backward(int i, cudaStream_t stream)
	{
		thrust::transform(binary_(i).begin(), binary_(i).end(), output.delta(i).begin(), input.delta(i).begin(), backward_kernel());
		return *this;
	}
}