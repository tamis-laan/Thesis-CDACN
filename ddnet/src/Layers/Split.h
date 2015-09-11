#pragma once
#include "Element.h"
#include "../Block.h"
#include <string>

namespace ddnet
{
	/*
		The Split takes a single Block as input and produces multiple output Blocks copying the input Block internal memory. This layer can be used to split blocks
		and feed the result into the next layer. Together with the MergeLayer we can pritty mutch construct arbitrary networks.
	*/
	class Split final : public Element
	{
	public:
		//Vector of Blocks required as output
		Block input;
		std::vector<Block> output;
		std::string method_;
	public:
		//Default Constructor
		Split() = default;
		//Constructor
		Split(cudnnHandle_t handle, std::string method = "sum");
		//Set input Blocks and output Block
		void set(Block &input, std::vector<Block> &output);
		//Set input Block
		void set_in(Block &block);
		//Set output Block
		void set_out(std::vector<Block> &blocks);
		//Split the input Block into multiple output Blocks
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Propegate back the delta's by taking the average delta of the outputs
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
	};
	//Helper function that makes things easier to work with
	inline void link(Block &input, Split &layer, std::vector<Block> output) {layer.set(input,output);}
}