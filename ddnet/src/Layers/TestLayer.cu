#include "TestLayer.h"


namespace ddnet
{
	struct avrg_kernel
	{
		const float decay;
		const float epsilon;
		avrg_kernel(float decay, float epsilon) : decay(decay), epsilon(epsilon) {}

		template <typename Tuple> 
		__device__ void operator()(Tuple t)
		{
			//Average delta
			thrust::get<1>(t) = (1.0-decay)*thrust::get<0>(t)*thrust::get<0>(t) + decay*thrust::get<1>(t);
			//Set new delta
			thrust::get<2>(t) = thrust::get<0>(t)/(sqrt(thrust::get<1>(t))+epsilon);
		}
	};

	Element& TestLayer::backward(int i, cudaStream_t stream)
	{
		//Just do nothing for now
		// copy(output.delta(i),input.delta(i));

		//apply RMSPROP kernel
		thrust::for_each(	thrust::make_zip_iterator(thrust::make_tuple(output.delta(i).begin(), delta_avrg(i).begin(), input.delta(i).begin() ) ),
							thrust::make_zip_iterator(thrust::make_tuple(output.delta(i).end(),   delta_avrg(i).end(),   input.delta(i).end()   ) ),
							avrg_kernel(0.98,0.01));
		return *this;
	}
}
