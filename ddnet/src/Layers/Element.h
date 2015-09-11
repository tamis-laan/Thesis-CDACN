#pragma once
#include "../Tensor.h"
#include "../Util.h"
#include <vector>

namespace ddnet
{
	/*
		Base class for all types of layers. This base class is used by the Network class in order to move forward and backwards through
		all the layers. It also keeps track of the handle, something that all classes require. The save and load functions will save and load 
		the element as a string. The idea being that the Network class will use these functions to save the network, while seperatly keeping track
		of the topology of the network.

		THE FOLLOWING PARAMETERS MUST BE SET BY DERIVED CLASSES:
		- bool guard : Set to true if everything is set correctly
	*/ 
	class Element
	{
	protected:
		//Handle associated with device (different handle different GPU)
		cudnnHandle_t handle = 0;
		//The guard is there to check if everything is set correctly
		bool guard_ = false;
	public:
		//Default constructor
		Element() = default;
		//Constructor
		Element(cudnnHandle_t handle) : handle(handle) {}
		//Get guard value
		bool guard() {return guard_;}
		//Compute the output given the input
		virtual Element& forward (int i, cudaStream_t stream = NULL) {return *this;};
		//Given the output delta compute the input delta
		virtual Element& backward(int i, cudaStream_t stream = NULL) {return *this;}
		//Given the output delta compute internel derivatives
		virtual Element& update  (int i, bool accumulate = false, cudaStream_t stream = NULL) {return *this;}
		//This function returns the weights that can be optimized
		virtual std::vector<Tensor> weights() {return std::vector<Tensor>();}
		//This function returns the gradients used to optimize weights
		virtual std::vector<Tensor> gradients() {return std::vector<Tensor>();}
		//Save the element as in the form of a string
		virtual std::string save() {return "";}
		//Load the element from a string
		virtual void load(std::string description) {};
	};
}