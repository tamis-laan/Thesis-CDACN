// #pragma once
// #include "ActivationLayer.h"
// #include <vector>
// #include <tuple>
// #include <thrust/device_vector.h>

// namespace ddnet
// {
// 	class Constraint final : public ActivationLayer 
// 	{
// 	private:
// 		std::vector<std::pair<float,float>> domain_;
// 		thrust::device_vector<float> lower_bound;
// 		thrust::device_vector<float> upper_bound;
// 	public:
// 		//Constructor
// 		Constraint(cudnnHandle_t handle, std::vector<std::pair<float,float>> domain);
// 		//Set output Block
// 		virtual void set_out(Block &block) override;
// 		//Forward data
// 		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
// 		//Backward data
// 		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
// 	};
// }