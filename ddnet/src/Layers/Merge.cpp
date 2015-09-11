#include "Merge.h"

namespace ddnet
{
	//Constructor
	Merge::Merge(cudnnHandle_t handle) : Element(handle) {}

	//Set input and output Blocks
	void Merge::set(std::vector<Block> &input, Block &output)
	{
		set_in(input);
		set_out(output);
		guard_ = true;
	}

	//Set input Block
	void Merge::set_in(std::vector<Block> &blocks)
	{
		//Check input concistency
		if(blocks.size()<2) throw ddnet_exception(2);
		for(int i=1; i<blocks.size(); i++)
			if(blocks[i-1].n() != blocks[i].n()) throw ddnet_exception(0);
		//If correct set as input
		input = blocks;
	}

	//Set output Block
	void Merge::set_out(Block &block)
	{
		//Check output concistency
		int sum = 0;
		for(int i=0; i<input.size(); i++)
			sum+=input[i].c()*input[i].h()*input[i].w();
		if(sum!=block.c()*block.h()*block.w()) throw ddnet_exception(0); 
		//If correct set as output
		output = block;
	}

	//Merge the input Blocks into the output Block
	Element& Merge::forward(int i, cudaStream_t stream)
	{
		//Shift will increase based on processed c*h*w
		int shift = 0;
		//First we compute the total c*h*w
		int size = 0;
		for(int k=0; k<input.size(); k++)
			size+=input[k].c()*input[k].h()*input[k].w();
		//Next we transform input into output
		for(int k=0; k<input.size() ;k++)
		{
			//Compute the stride, we need the total c*h*w except for our own c*h*w
			int stride = size - input[k].c()*input[k].h()*input[k].w();
			//Transform data
			transform_src(input[k].data(i),output.data(i),shift,stride);
			//Increase shift
			shift+=input[k].c()*input[k].h()*input[k].w();
		}
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}

	//Split the output Block delta into the input Block's delta's
	Element& Merge::backward(int i, cudaStream_t stream)
	{
		//Shift will increase based on processed c*h*w
		int shift = 0;
		//First we compute the total c*h*w
		int size = 0;
		for(int k=0; k<input.size(); k++)
			size+=input[k].c()*input[k].h()*input[k].w();
		//Next we transform input into output
		for(int k=0; k<input.size() ;k++)
		{
			//Compute the stride, we need the total c*h*w except for our own c*h*w
			int stride = size - input[k].c()*input[k].h()*input[k].w();
			//Transform data
			transform_dst(output.delta(i),input[k].delta(i),shift,stride);
			//Increase shift
			shift+=input[k].c()*input[k].h()*input[k].w();
		}
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}
}