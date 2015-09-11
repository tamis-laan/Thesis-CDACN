#include "ADAM.h"

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

namespace ddnet
{
	struct adam_kernel
	{
		const float learning_rate;
		const float decay_1;
		const float decay_2;
		const float epsilon;
		adam_kernel(float learning_rate, float decay_1, float decay_2, float epsilon) : learning_rate(learning_rate), decay_1(decay_1), decay_2(decay_2), epsilon(epsilon) {}

		template <typename Tuple>
		__device__ void operator()(Tuple t)
		{
			//thrust::get<0>(t) = gradients
			//thrust::get<1>(t) = buffer 1
			//thrust::get<2>(t) = buffer 2
			//thrust::get<3>(t) = weights

			//Decay buffer 1
			thrust::get<1>(t) = (1.0-decay_1)*thrust::get<0>(t) + decay_1*thrust::get<1>(t);
			//Decay buffer 2
			thrust::get<2>(t) = (1.0-decay_2)*thrust::get<0>(t) * thrust::get<0>(t) + decay_2*thrust::get<2>(t);
			//Set weights
			thrust::get<3>(t) = thrust::get<3>(t) + learning_rate*sqrt(1.0-decay_2)/(1.0-decay_1) * thrust::get<1>(t)/(sqrt(thrust::get<2>(t))+epsilon);
		}
	};

	void ADAM::step(Adapter &adaptor, cudaStream_t stream)
	{
		for(int i=0; i<adaptor.size(); i++)
		{
			//apply ADAM kernel
			thrust::for_each(	thrust::make_zip_iterator(thrust::make_tuple(adaptor.gradients(i).begin(), adaptor.buffers(0,i).begin(), adaptor.buffers(1,i).begin(), adaptor.weights(i).begin() ) ),
								thrust::make_zip_iterator(thrust::make_tuple(adaptor.gradients(i).end(),   adaptor.buffers(0,i).end(),   adaptor.buffers(1,i).end(),   adaptor.weights(i).end()   ) ),
								adam_kernel(meta_parameters_[0],meta_parameters_[1],meta_parameters_[2], meta_parameters_[3]));
		}
	}
}
	