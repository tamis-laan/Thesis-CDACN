#pragma once
#include "Element.h"
#include "../Block.h"
namespace ddnet
{
	/*
		Standaard layer class which takes one input and one output. This class is ment to be overidden in order to add a new layer
		to the system. The set, set_in and set_out methods differ per layer. Depending on allocation of different resources.
	*/
	class Layer : public Element
	{
	protected:
		//Input output layer blocks 
		Block input;
		Block output;
	public:
		//Default constuctor
		Layer() = default;
		//Constructor sets handle
		Layer(cudnnHandle_t handle) : Element(handle) {}
		//Used to set input and ouput Blocks
		virtual void set(Block &input, Block &output) final {set_in(input); set_out(output); guard_ = true;}
		//Used to set input Block
		virtual void set_in (Block &block) = 0;
		//Used to set output Block
		virtual void set_out(Block &block) = 0;
	};
	//Helper function that makes things easier to work with
	inline void link(Block &input, Layer &layer, Block &output) {layer.set(input,output);}
}