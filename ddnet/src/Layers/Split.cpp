#include "Split.h"
#include <algorithm>

namespace ddnet
{

	//Constructor
	Split::Split(cudnnHandle_t handle, std::string method) : Element(handle), method_(method) {}


	//Set input Blocks and output Block
	void Split::set(Block &input, std::vector<Block> &output)
	{
		set_in(input);
		set_out(output);
		guard_ = true;
	}

	//Set input Block
	void Split::set_in(Block &block)
	{
		input = block;
	}

	//Set output Block
	void Split::set_out(std::vector<Block> &blocks)
	{
		//Check concistency
		for(int i=0; i<blocks.size(); i++)
			if(blocks[i].n() != input.n()) throw ddnet_exception(0);
		for(int i=0; i<blocks.size(); i++)
			if(blocks[i].c()*blocks[i].h()*blocks[i].w() != input.c()*input.h()*input.w()) throw ddnet_exception(0);
		//If correct set as output
		output = blocks;
	}

	//Split the input Block into multiple output Blocks
	Element& Split::forward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Copy over input into the outputs
		for(int k=0; k<output.size(); k++)
			copy(input.data(i),output[k].data(i));
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}

	//Propegate back the delta's by taking the average delta of the outputs
	Element& Split::backward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Null out the input delta
		cudnn_fill(0.0,input.delta(i));
		//Average the output delta's into the input
			if(method_ == "sum")
				for(int k=0; k<output.size(); k++)
					cudnn_add(1.0,output[k].delta(i),1.0,input.delta(i)); //NOTE: Might not be the correct way
			if(method_ == "average")
				for(int k=0; k<output.size(); k++)
					cudnn_add(1.0/output.size(),output[k].delta(i),1.0,input.delta(i));
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	}
}