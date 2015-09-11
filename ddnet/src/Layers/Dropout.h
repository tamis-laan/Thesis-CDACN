#pragma once
#include "Layer.h"

namespace ddnet
{
	/*
		This class applies Dropout to the input. That with a given probability each input will be set to zero in the forward pass. 
		There are 2 modes, dropout and prediction. When prediction is true instead of dropout being applied the input is mulitplied
		with the dropout probability. This is done if we want to predict instead of train on data. For which data stream to use
		dropout or predict is set using a bool vector. True means predict and false means dropout. If no bool vector is given 
		then by default data streams with only one image (batch size 1) will be set to true, i.e. prediction.
	*/
	class Dropout final : public Layer
	{
	protected:
		//Resulting vector
		std::vector<bool> p_;
		//Binary vector set using threshold probabillity
		TensorArray binary_;
		//Threshold probability
		float prob_;
	public:
		Dropout() = default;
		Dropout(cudnnHandle_t handle, float prob = 0.5, std::vector<bool> p = std::vector<bool>());
		//Set input Block
		virtual void set_in(Block &block) override;
		//Set output Block
		virtual void set_out(Block &block) override;
		//Forward data
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Backward data
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
	};
}