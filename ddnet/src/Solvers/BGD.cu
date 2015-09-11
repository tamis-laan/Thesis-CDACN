#include "BGD.h"

namespace ddnet
{
	struct bgd_kernel
	{
		const float learning_rate;
		const float momentum;
		bgd_kernel(float learning_rate, float momentum) : learning_rate(learning_rate), momentum(momentum) {}

	    template <typename Tuple>
	    __device__ void operator()(Tuple t)
	    {
	    	//thrust::get<0>(t) = gradients
	    	//thrust::get<1>(t) = momentum buffer
	    	//thrust::get<2>(t) = weights
	    	thrust::get<1>(t) = thrust::get<0>(t) + momentum*thrust::get<1>(t);
	    	thrust::get<2>(t) = thrust::get<2>(t) + learning_rate * thrust::get<1>(t);
	    }
	};

	void BGD::step(Adapter &adaptor, cudaStream_t stream)
	{
		for(int i=0; i<adaptor.size(); i++)
		{
			//apply Batch Gradient Descent kernel
			thrust::for_each(	thrust::make_zip_iterator(thrust::make_tuple(adaptor.gradients(i).begin(), adaptor.buffers(0,i).begin(), adaptor.weights(i).begin() ) ),
								thrust::make_zip_iterator(thrust::make_tuple(adaptor.gradients(i).end(),   adaptor.buffers(0,i).end(),   adaptor.weights(i).end()   ) ),
								bgd_kernel(meta_parameters_[0],meta_parameters_[1]));
		}
	}
}