#pragma once
#include "../src/Network.h"

#include <vector>
#include <tuple>

using namespace ddnet;
using namespace std;
/*
	Example of how to construct a network. This example defines the MNIST network that takes images of hand written digits from the MNIST data set
	and learns to recognice them. As you can see the network is defined internally.
*/
class MNIST : public Network
{
public:
	MNIST(cudnnHandle_t handle, std::vector<int> batch) : Network(handle,batch)
	{
		// //Create Network topology
		// 	auto b1 = create<Block>(1,28,28);
		// 	auto l1 = create<ConvolutionLayer>(16,8,8,4,4,0.01);
		// 	auto b2 = create<Block>(16,6,6);
		// 	auto l2 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		// 	auto b3 = create<Block>(16,6,6);
		// 	auto l3 = create<ConvolutionLayer>(32,4,4,2,2,0.01);
		// 	auto b4 = create<Block>(32,2,2);
		// 	auto l4 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		// 	auto b5 = create<Block>(32,2,2);
		// 	auto l5 = create<DenseLayer>(256,0.01);
		// 	auto b6 = create<Block>(256);
		// 	auto l6 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		// 	auto b7 = create<Block>(256);
		// 	auto l7 = create<DenseLayer>(1,0.01);
		// 	auto b8 = create<Block>(1);
		// 	//Create Network Streams
		// 	auto s1 = create<Stream>(BGD(),b1,l1,b2,l2,b3,l3,b4,l4,b5,l5,b6,l6,b7,l7,b8);
		// 	//Push stream
		// 	push(s1);

		//Create Network topology
			auto b1 = create<Block>(1,28,28);
			auto sl = create<SplitLayer>();

			auto a  = create<Block>(1,28,28);
			auto b  = create<Block>(1,28,28);

			auto l1 = create<ConvolutionLayer>(16,8,8,4,4,0.01);
			auto b2 = create<Block>(16,6,6);
			auto l2 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			auto b3 = create<Block>(16,6,6);
			auto l3 = create<ConvolutionLayer>(32,4,4,2,2,0.01);
			auto b4 = create<Block>(32,2,2);
			auto l4 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			auto b5 = create<Block>(32,2,2);
			auto l5 = create<DenseLayer>(256,0.01);
			auto b6 = create<Block>(256);
			auto l6 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			auto b7 = create<Block>(256);
			auto l7 = create<DenseLayer>(1,0.01);
			auto b8 = create<Block>(1);
			//Create Network Streams
			auto s1 = create<Stream>(ADAM(),b1,sl,vector<Block>{a,b});
			auto s2 = create<Stream>(ADAM(),a,l1,b2,l2,b3,l3,b4,l4,b5,l5,b6,l6,b7,l7,b8);
			//Push Streams
			push(s1);
			push(s2);

		//Create Network topology
			// auto b1 = create<Block>(1,28,28);
			// auto sl = create<SplitLayer>();

			// auto a  = create<Block>(1,28,28);
			// auto b  = create<Block>(1,28,28);

			// auto l1 = create<DenseLayer>(64,1.0);
			// auto b2 = create<Block>(64);
			// auto l2 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			// auto b3 = create<Block>(64);
			// auto l3 = create<DenseLayer>(128);
			// auto b4 = create<Block>(128);
			// auto l4 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			// auto b5 = create<Block>(128);
			// auto l5 = create<DenseLayer>(256,1.0);
			// auto b6 = create<Block>(256);
			// auto l6 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
			// auto b7 = create<Block>(256);
			// auto l7 = create<DenseLayer>(1,1.0);
			// auto b8 = create<Block>(1);
			// //Create Network Streams
			// auto s1 = create<Stream>(ADAM(),b1,sl,vector<Block>{a,b});
			// auto s2 = create<Stream>(ADAM(),a,l1,b2,l2,b3,l3,b4,l4,b5,l5,b6,l6,b7,l7,b8);
			// //Push Streams
			// push(s1);
			// push(s2);

		// //Create Network topology
		// 	auto b1 = create<Block>(1,28,28);
		// 	auto l1 = create<DenseLayer>(64,0.01);
		// 	auto b2 = create<Block>(64);
		// 	auto l2 = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		// 	auto b3 = create<Block>(64);
		// 	auto l3 = create<DenseLayer>(1,0.01);
		// 	auto b4 = create<Block>(1);
		// 	//Create Network Streams
		// 	auto s = create<Stream>(ADAM(0.0001,0.9,0.9,1e-12),b1,l1,b2,l2,b3,l3,b4);
		// 	//Push Streams
		// 	push(s);
	
		//Set Inputs 
		inputs(b1);
		//Set Outputs
		outputs(b8);

	}
};

/*
	Testing test Layer
*/
class MNIST_avrg : public Network
{
public:
	MNIST_avrg(cudnnHandle_t handle, std::vector<int> batch) : Network(handle,batch)
	{
		//Create Network topology
		auto b1  = create<Block>(1,28,28);
		auto tl1 = create<TestLayer>();
		auto tb1 = create<Block>(1,28,28);
		auto l1  = create<ConvolutionLayer>(16,8,8,4,4,0.01);
		auto b2  = create<Block>(16,6,6);
		auto tl2 = create<TestLayer>();
		auto tb2 = create<Block>(16,6,6);
		auto l2  = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		auto b3  = create<Block>(16,6,6);
		auto tl3 = create<TestLayer>();
		auto tb3 = create<Block>(16,6,6);
		auto l3  = create<ConvolutionLayer>(32,4,4,2,2,0.01);
		auto b4  = create<Block>(32,2,2);
		auto tl4 = create<TestLayer>();
		auto tb4 = create<Block>(32,2,2);
		auto l4  =  create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		auto b5  = create<Block>(32,2,2);
		auto tl5 = create<TestLayer>();
		auto tb5 = create<Block>(32,2,2);
		auto l5  =  create<DenseLayer>(256,0.01);
		auto b6  = create<Block>(256);
		auto tl6 = create<TestLayer>();
		auto tb6 = create<Block>(256);
		auto l6  = create<ActivationLayer>(CUDNN_ACTIVATION_RELU);
		auto b7  = create<Block>(256);
		auto tl7 = create<TestLayer>();
		auto tb7 = create<Block>(256);
		auto l7  = create<DenseLayer>(1,0.01);
		auto b8  = create<Block>(1);
		auto tl8 = create<TestLayer>();
		auto tb8 = create<Block>(1);
		//Create Network Streams
		auto s1 = create<Stream>(BGD(),b1,tl1,tb1,l1,b2,tl2,tb2,l2,b3,tl3,tb3,l3,b4,tl4,tb4,l4,b5,tl5,tb5,l5,b6,tl6,tb6,l6,b7,tl7,tb7,l7,b8,tl8,tb8);
		//Push Streams
		push(s1);
		//Set Inputs 
		inputs(b1);
		//Set Outputs
		outputs(tb8);
	}
};