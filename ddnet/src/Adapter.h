#pragma once
#include "./Layers/Layer.h"
#include "./Layers/Convolution.h"
#include "./Layers/Dense.h"
#include "./Layers/Activation.h"
#include "./Layers/Pool.h"

namespace ddnet
{
	/*
		This class wraps around an Element and manages buffers used by the Solver class. These buffers are used to store
		temporary values such as the momentum for BGD. (NOTE: We could add streams to the adaptor, then we would have 1 stream per element)
	*/
	class Adapter final
	{
	private:
		//Element itself
		std::shared_ptr<Element> element_;
		//Pointer to element weights (GPU)
		std::vector<Tensor> weights_;
		//Pointer to element gradients (GPU)
		std::vector<Tensor> gradients_;
		//Beuffers size of weights for solver (GPU)
		std::vector<std::vector<Tensor>> buffers_;
	public:
		//Default Constructor
		Adapter() = default;
		//Constructor
		Adapter(std::shared_ptr<Element> element, std::vector<float> buffer_values);
		//Return the number of weights
		int size();
		//Return elements
		Element& element();
		//Return parameters
		Tensor& weights(int i);
		//Return gradients
		Tensor& gradients(int i);
		//Return buffers
		std::vector<Tensor>& buffers(int i);
		//Return buffers
		Tensor& buffers(int i, int j);
		//Transfer weights to other Adapter
		void transfer(Adapter &a);
	};
}