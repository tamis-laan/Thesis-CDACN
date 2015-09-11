#pragma once

#include "./Block.h"
#include "./Adapter.h"
#include "./Solvers/Solver.h"
#include "./Solvers/RMSPROP.h"
#include "./Solvers/SMORMS3.h"
#include "./Solvers/ADAM.h"
#include "./Solvers/BGD.h"
#include "./Layers/Convolution.h"
#include "./Layers/Dense.h"
#include "./Layers/Activation.h"
#include "./Layers/Constraint.h"
#include "./Layers/Pool.h"
#include "./Layers/Dropout.h"
#include "./Layers/Merge.h"
#include "./Layers/Split.h"
#include "./Layers/TestLayer.h"

#include <memory>

namespace ddnet
{
	/*
		The stream Containt encapsulates a linear sequence of elements that that have a linear dependence in terms of execution. In other words each layer is dependent on
		the preveus layer and cannot be executed in parallel. On the other hand mutliple streams can be executed in parallel by the Network class. 
	*/
	class Stream
	{
	private:
		std::shared_ptr<Solver> solver_;
		std::vector<Adapter> adapters_;
	public:
		//Constructor
		Stream(std::shared_ptr<Solver> solver, std::vector<Adapter> adapters);
		//Get the solver
		std::shared_ptr<Solver> getSolver();
		//Forward input batch trough stream
		void forward(int batch);
		//Backward stream
		void backward(int batch);
		//Update Stream
		void update(int batch);
		//Transfer weights between streams
		void transfer(Stream &s);
	};

	/*
		The Network class stores the Streams, Blocks and Elements. It represents a complete usable Neural Network. 
	*/
	class Network
	{

	private:
		//Device handle
		cudnnHandle_t handle_;
		//Batch sizes
		std::vector<int> batch_;
		//Streams
		std::vector<std::vector<Stream>> streams_;
		//Blocks 
		std::vector<Block> blocks_;
		//Inputs
		std::vector<Block> inputs_;
		//Outputs
		std::vector<Block> outputs_;
		//Base Case Block
		std::vector<Adapter> generate_adapter_list(std::vector<float> buffer_values, Block& block) { return {}; }
		//Base Case Vector Block
		std::vector<Adapter> generate_adapter_list(std::vector<float> buffer_values, std::vector<Block> &block) { return {}; }
		//Induction case
		template<typename F, typename E, typename S, typename... Args> std::vector<Adapter> generate_adapter_list(std::vector<float> buffer_values, F &first, E &element, S &second, Args&&... args)
		{
			//Link triplet together
			link(first,element,second);
			//Build Adapter
			Adapter adapter(std::make_shared<E>(element), buffer_values);
			//Build Adapter vector
			std::vector<Adapter> adapters = {adapter};
			//Recurse
			auto adapters_tail = generate_adapter_list(buffer_values,second,args...);
			//Concatenate Adapter vectors
			adapters.insert(adapters.end(),adapters_tail.begin(),adapters_tail.end());
			//Return adaptor vector
			return adapters;
		}
	public:
		//Default Constructor
		Network() = default;
		//Constructor
		Network(cudnnHandle_t handle, std::vector<int> batch);

		//Create a Element
		template <typename T, typename... Args, typename std::enable_if<std::is_base_of<Element, T>::value>::type* = nullptr> T create(Args&&... args) 
		{
			return T(handle_, std::forward<Args>(args)...);
		}

		//Create a Block
		template <typename T, typename... Args, typename std::enable_if<std::is_same<Block, T>::value>::type* = nullptr> T create(Args&&... args) 
		{
			Block block(handle_, batch_, std::forward<Args>(args)...);
			blocks_.push_back(block);
			return block;
		}

		//Create a Stream
		template <typename T, typename S, typename... Args, typename std::enable_if<std::is_same<Stream, T>::value>::type* = nullptr> T create(S solver, Args&&... args) 
		{
			return Stream( std::shared_ptr<Solver>(new S(solver)), generate_adapter_list(solver.buffer_values(), args...) );
		}

		//Clone a Block/Element
		template<typename T> T clone(T &object)
		{
			static_assert(std::is_base_of<Element,T>::value or std::is_same<Block,T>::value, "Not Block/Element!");
			return T(object);
		}

		//Push streams
		template<typename... Args> void push(Args&&... args)
		{
			streams_.push_back({args...});
		}

		//Set input blocks
		template<typename... Args> void inputs(Args&&... args)
		{
			inputs_ = {args...};
		}

		//Set output blocks
		template<typename... Args> void outputs(Args&&... args)
		{
			outputs_ = {args...};
		}

		//Get stream refrence
		Stream& stream(int i, int j);
		//Get an input block
		Block& input(int i);
		//Get an output block
		Block& output(int i);
		//Forward streams
		void forward(int batch);
		//Backward streams
		void backward(int batch);
		//Update streams
		void update(int batch);
		//Transfer weights to identical network
		void transfer(Network &n);
	};

}