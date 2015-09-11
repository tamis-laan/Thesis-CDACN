#pragma once
#include "Element.h"
#include "../Block.h"

namespace ddnet
{
	/*
		The Merge takes multiple Blocks as input and produces one Block as output. This layer can be used to merge blocks together
		and feed the result into the next layer. Together with the SplitLayer we can pritty mutch construct arbitrary networks.
	*/
	class Merge final : public Element
	{
	protected:
		//Vector of blocks required as input
		std::vector<Block> input;
		Block output;
	public:
		//Default constructor
		Merge() = default;
		//Constructor
		Merge(cudnnHandle_t handle);
		//Set input and output Blocks
		void set(std::vector<Block> &input, Block &output);
		//Set input Block
		void set_in(std::vector<Block> &blocks);
		//Set output Block
		void set_out(Block &block);
		//Merge the input Blocks into the output Block
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Split the output Block delta into the input Block's delta's
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
	};
	//Helper function that makes things easier to work with
	inline void link(std::vector<Block> input, Merge &layer, Block &output) { layer.set(input,output); }
}