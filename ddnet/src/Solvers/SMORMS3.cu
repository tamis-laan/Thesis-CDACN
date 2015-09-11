#include "SMORMS3.h"

#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>

namespace ddnet
{
	struct smorms3_kernel
	{
		const float learning_rate;
		const float epsilon;
		smorms3_kernel(float learning_rate, float epsilon) : learning_rate(learning_rate), epsilon(epsilon) {}
		template <typename Tuple> __device__ void operator()(Tuple t)
		{
			float r =  1.0/(thrust::get<4>(t)+1.0);
			thrust::get<2>(t) = (1.0-r)*thrust::get<2>(t) + r*thrust::get<1>(t);
			thrust::get<3>(t) = (1.0-r)*thrust::get<3>(t) + r*thrust::get<1>(t)*thrust::get<1>(t);
			thrust::get<0>(t) = thrust::get<0>(t) + thrust::get<1>(t)*min(learning_rate,thrust::get<2>(t)*thrust::get<2>(t)/(thrust::get<3>(t)+epsilon))/(sqrt(thrust::get<3>(t))+epsilon);
			// thrust::get<0>(t) = thrust::get<0>(t) + thrust::get<1>(t)*thrust::get<2>(t)*thrust::get<2>(t)/(thrust::get<3>(t)+epsilon)/(sqrt(thrust::get<3>(t))+epsilon);
			thrust::get<4>(t) = 1.0+thrust::get<4>(t)*(1.0-thrust::get<2>(t)*thrust::get<2>(t)/(thrust::get<3>(t)+epsilon));
		}
	};

	void SMORMS3::step(Adapter &adaptor, cudaStream_t stream)
	{
		for(int i=0; i<adaptor.size(); i++)
		{
			//apply SMORMS3 kernel
			thrust::for_each(	thrust::make_zip_iterator(thrust::make_tuple(adaptor.weights(i).begin(), adaptor.gradients(i).begin(), adaptor.buffers(0,i).begin(), adaptor.buffers(1,i).begin(), adaptor.buffers(2,i).begin()  ) ),
								thrust::make_zip_iterator(thrust::make_tuple(adaptor.weights(i).end(),   adaptor.gradients(i).end(),   adaptor.buffers(0,i).end(),   adaptor.buffers(1,i).end(),   adaptor.buffers(2,i).end()    ) ),
								smorms3_kernel(meta_parameters_[0],meta_parameters_[1]));
		}
	}	
}