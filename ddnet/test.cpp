#include "thrust/device_vector.h"
#define CATCH_CONFIG_MAIN 

#include <catch.h>
#include <thread>
#include <future>

#include "./src/Tensor.h"
#include "./src/TensorArray.h"
#include "./src/Block.h"
#include "./src/Layers/ConvolutionLayer.h"
#include "./src/Layers/DenseLayer.h"
#include "./src/Layers/ActivationLayer.h"
#include "./src/Layers/ConstraintLayer.h"
#include "./src/Layers/MergeLayer.h"
#include "./src/Layers/SplitLayer.h"
#include "./src/Layers/PoolingLayer.h"
#include "./src/Layers/DropoutLayer.h"
#include "./src/Adapter.h"
#include "./src/Solvers/Solver.h"
#include "./src/Network.h"
#include "./src/Solvers/BGD.h"
#include "./networks/MNIST.h"
#include "./src/Solvers/ADAM.h"

using namespace std;
using namespace ddnet;

TEST_CASE( "Tensor test", "[Tensor]" ) 
{
    //Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructors & assgnment operators
	Tensor tensor1;
	{
		Tensor tmp(handle,1,1,2,2);
		tensor1 = tmp;
	}
	Tensor tensor2 = Tensor(handle,1,1,2,2);

	//Flat style tensor
	Tensor tensor3 = Tensor(handle,1,5);

	//Check if default value is correct
	REQUIRE(tensor1(0,0)[0] == 0.0);

	//Test element access trough device_refrence
	tensor1(0,0,0,0) = 1.0; tensor1(0,0,0,1) = 0.0;
	tensor2(0,0,0,0) = 2.0; tensor2(0,0,0,1) = 0.0;

	//Test add
	cudnn_add(2.0,tensor1,1.0,tensor2);

	REQUIRE(tensor1(0,0,0,0) == 1.0); 
	REQUIRE(tensor1(0,0,0,1) == 0.0);

	REQUIRE(tensor2(0,0,0,0) == 4.0); 
	REQUIRE(tensor2(0,0,0,1) == 0.0);

	//Test scale
	cudnn_scale(3.0,tensor1);
	REQUIRE(tensor1(0,0,0,0) == 3.0); 
	REQUIRE(tensor1(0,0,0,1) == 0.0);

	//Test fill
	cudnn_fill(31,tensor2);
	REQUIRE(tensor2(0,0,0,0) == 31);
	REQUIRE(tensor2(0,0,0,1) == 31);

	//Test reshaping tensor
	tensor1.reshape(1,1,1,4);
	cudnnDataType_t type; int n; int c; int h; int w; int ns; int cs; int hs; int ws;
	check( cudnnGetTensor4dDescriptor(tensor1.descriptor(),&type,&n,&c,&h,&w,&ns,&cs,&hs,&ws) );
	REQUIRE(n==1);
	REQUIRE(c==1); 
	REQUIRE(h==1); 
	REQUIRE(w==4);

	//Test random number generation
	curand_rand_normal(0.0,2.0,tensor1);

	//Test pointer access through device_ptr
	tensor1(0,0)[0] = 1.0; tensor1(0,0)[1] = 2.0;
	tensor1(0,0)[2] = 3.0; tensor1(0,0)[3] = 4.0;

	REQUIRE(tensor1(0)[0] == 1.0); 
	REQUIRE(tensor1(0)[3] == 4.0);

	//Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "TensorArray test", "[TensorArray]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructors & assgnment operators
	TensorArray array1;
	{
		TensorArray tmp(handle,{3,4},3,32,32);
		array1 = tmp;
	}
	TensorArray array2 = TensorArray(handle,{3,4,7},1,8,8);

	//Flat style TensorArray
	TensorArray array3 = TensorArray(handle,{3,4,7},64);
	
	//Test tensor array access
	array1(0,2,2)[32*32-1] = 3.0;
	REQUIRE(array1(0,2,2,31,31) == 3.0);
	
	//Test reshaping
	array2.reshape({3,4,7},1,1,64);

	//Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "Block test", "[Block]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructors & assgnment operators
	Block block1;
	{
		Block tmp(handle,{3,4},3,32,32);
		block1 = tmp;
	}
	Block block2 = Block(handle,{3,4,7},1,8,8);

	//Flat style Block
	Block block3 = Block(handle,{3,4,7},64);

	//Test block tensor array access
	block1.data(0,2,2)[32*32-1] = 3.0;
	REQUIRE(block1.data(0,2,2,31,31) == 3.0);

	//Test Block reshaping
	block2.reshape({3,4,7},1,1,64);

	//Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "ConvolutionLayer test", "[ConvolutionLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test Filter constructors & assgnment operators
	Filter filter1;
	{
		Filter tmp(handle,1,1,2,2);
		filter1 = tmp;
	}
	Filter filter2 = Filter(handle,1,1,2,2);

	//Test [] operator
	filter1[0] = 1.0; filter1[1] = 0.0;
	filter2[0] = 2.0; filter2[1] = 0.0;

	//Test add
	cudnn_add(2.0,filter1,1.0,filter2);
	REQUIRE(filter1[0] == 1.0);  
	REQUIRE(filter1[1] == 0.0);
	REQUIRE(filter2[0] == 4.0);
	REQUIRE(filter2[1] == 0.0);

	//Test constructing the convolution layer
	Block            b1(handle,{1},1,1,2);
	ConvolutionLayer l2(handle,1,1,2);
	Block            b3(handle,{1},1,1,1);

	//Test set method
	link(b1,l2,b3);

	//Test forward 
	b1.data(0,0,0)[0] = 2.0;  b1.data(0,0,0)[1] = 3.0;
	l2.filter()[0] = 3.0;     l2.filter()[1] = -4.0;     l2.bias()[0] = 1.0;
	
	l2.forward(0);

	REQUIRE(b3.data(0)[0] == 2);

	//Test backward 
	Tensor delt = b3.delta(0);
	Tensor DyDx = b1.delta(0);
	cudnn_fill(1.0,delt);

	l2.backward(0);

	REQUIRE(DyDx[0] == -4.0); 
	REQUIRE(DyDx[1] ==  3.0);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "DenseLayer 1 test", "[DenseLayer 1]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block      b1(handle,{1},1);
	DenseLayer l2(handle,1);
	Block      b3(handle,{1},1);

	//Test set method
	link(b1,l2,b3);

	//Set the filter/bias and input
	b1.data(0,0,0)[0] = 2.0;
	l2.filter()[0]    = 3.0;
	l2.bias()[0]      = 2.0;
	b3.delta(0)[0]    = 1.0;

	//Step
	l2.forward(0);
	l2.backward(0);
	l2.update(0);

	//Check values
	REQUIRE(b3.data(0)[0]        == 8.0); //output
	REQUIRE(b1.delta(0)[0]       == 3.0); //d/dx
	REQUIRE(l2.filter_delta()[0] == 2.0); //d/dw
	REQUIRE(l2.bias_delta()[0]   == 1.0); //d/db
	
	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "DenseLayer test 2", "[DenseLayer 2]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block      b1(handle,{1},2);
	DenseLayer l2(handle,1);
	Block      b3(handle,{1},1);

	//Test set method
	link(b1,l2,b3);

	//Set the filter/bias and input
	b1.data(0,0,0)[0] = 2.0;  b1.data(0,0,0)[1] = 3.0;
	l2.filter()[0] = 3.0;     l2.filter()[1] = -4.0;     l2.bias()[0] = 1.0;

	//Test forward 
	l2.forward(0);

	//Check output
	REQUIRE(b3.data(0)[0] == 2); //output

	//Set delta values
	Tensor delt = b3.delta(0);
	Tensor DyDx = b1.delta(0);
	cudnn_fill(1.0,delt);

	//Test backward 
	l2.backward(0);

	//Check derivatives
	REQUIRE(DyDx[0] == -4.0);
	REQUIRE(DyDx[1] ==  3.0); // d/dx

	l2.update(0);

	//Check weight derivatives
	REQUIRE(l2.gradients()[0][1] == 2.0);// d/dw1
	REQUIRE(l2.gradients()[0][0] == 3.0);// d/dw2
	REQUIRE(l2.gradients()[1][0] == 1.0);// d/db

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "ActivationLayer test", "[ActivationLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block           b1(handle,{1},2);
	DenseLayer      l2(handle,1);
	Block           b3(handle,{1},1);
	ActivationLayer l4(handle);
	Block           b5(handle,{1},1);

	//Test the set method
	link(b1,l2,b3); link(b3,l4,b5);

	//Test forward 
	b1.data(0,0,0)[0] = 2.0;  b1.data(0,0,0)[1] = 3.0;
	l2.filter()[0] = -3.0;    l2.filter()[1] = -4.0;     l2.bias()[0] = 1.0;

	l2.forward(0); 
	l4.forward(0);

	//Test backward 
	Tensor delt = b3.delta(0);
	Tensor DyDx = b1.delta(0);
	
	cudnn_fill(1.0,delt);
	
	l4.backward(0); 
	l2.backward(0);

	REQUIRE(DyDx[0] == 0.0);
	REQUIRE(DyDx[1] == 0.0);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "ConstraintLayer test", "[ConstraintLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block           b1(handle,{1},2);
	ConstraintLayer l2(handle,{{0.0,2.0},{1.0,3.0}});
	Block           b3(handle,{1},2);

	//Test set method
	link(b1,l2,b3);

	//Set the filter/bias and input
	b1.data(0,0,0)[0] = 0;  
	b1.data(0,0,0)[1] = 0;

	//Test forward 
	l2.forward(0);

	//Check output
	REQUIRE(b3.data(0)[0] == 1.0); 
	REQUIRE(b3.data(0)[1] == 2.0); 

	// Destroy handle
	cudnnDestroy(handle);
}


TEST_CASE( "MergeLayer test", "[MergeLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Setup merge laryer
	Block      b1(handle,{2},2);
	Block      b2(handle,{2},1);
	Block      b3(handle,{2},1);
	MergeLayer m4(handle);
	Block      b5(handle,{2},4);
	link({b1,b2,b3},m4,b5);

	//Set values
	cudnn_fill(1.0f,b1.data(0));
	cudnn_fill(2.0f,b2.data(0));
	cudnn_fill(3.0f,b3.data(0));

	//Forward
	m4.forward(0);
	m4.forward(0);

	//Test result
	REQUIRE(b5.data(0,0,0,0,0) == 1.0); 
	REQUIRE(b5.data(0,0,0,0,1) == 1.0); 
	REQUIRE(b5.data(0,0,0,0,2) == 2.0);
	REQUIRE(b5.data(0,0,0,0,3) == 3.0);

	//Set delta
	b5.delta(0,0,0,0,0) = 1.0;
	b5.delta(0,0,0,0,1) = 1.0;
	b5.delta(0,0,0,0,2) = 2.0;
	b5.delta(0,0,0,0,3) = 3.0;

	//Backward merge layer
	m4.backward(0);
	m4.backward(0);

	//Test result
	REQUIRE(b1.delta(0,0,0,0,0) == 1.0);
	REQUIRE(b1.delta(0,0,0,0,1) == 1.0);
	REQUIRE(b2.delta(0,0,0,0,0) == 2.0); 
	REQUIRE(b3.delta(0,0,0,0,0) == 3.0);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "SplitLayer test 1", "[SplitLayer1]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Setup the SplitLayer
	Block      b1(handle,{2},2);
	SplitLayer s2(handle);
	Block      b3(handle,{2},2);
	Block      b4(handle,{2},2);
	Block      b5(handle,{2},2);
	link(b1,s2,{b3,b4,b5});

	//Set value
	cudnn_fill(3.0f,b1.data(0));

	//Forward
	s2.forward(0);
	s2.forward(0);

	//Test result
	REQUIRE(b3.data(0,0,0,0,0)==3.0); 
	REQUIRE(b4.data(0,0,0,0,0)==3.0); 
	REQUIRE(b5.data(0,0,0,0,0)==3.0);

	//Set delta
	cudnn_fill(1.0f,b3.delta(0));
	cudnn_fill(2.0f,b4.delta(0));
	cudnn_fill(3.0f,b5.delta(0));

	//Backward split layer
	s2.backward(0);
	s2.backward(0);

	//Test result
	REQUIRE(b1.delta(0,0,0,0,0) == 6.0);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "SplitLayer test 2", "[SplitLayer1]" )
{
	//Require handle for operations	
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	{
		//Construct the DenseLayer form
		Block      b1(handle,{1},1);
		DenseLayer l1(handle,1);
		Block      b2(handle,{1},1);
		DenseLayer l2(handle,2);
		Block      b3(handle,{1},2);

		//Link
		link(b1,l1,b2);
		link(b2,l2,b3);

		//Set the filter/bias and input
		b1.data(0)[0]  = 1.0;
		l1.filter()[0] = 2.0;

		l2.filter()[0] = 1.0;
		l2.filter()[1] = 4.0;

		b3.delta(0)[0] = 1.0;
		b3.delta(0)[1] = 1.0;

		//f/b/u
		l1.forward(0);
		l2.forward(0);
		l2.backward(0);
		l1.backward(0);
		l2.update(0);
		l1.update(0);

		REQUIRE(b3.data(0)[0]  == 2 );
		REQUIRE(b3.data(0)[1]  == 8 );
		REQUIRE(b1.delta(0)[0] == 10);
	}
	{
		//Construct SplitLayer version
		Block      b1(handle,{1},1);
		DenseLayer l1(handle,1);
		Block      b2(handle,{1},1);
		SplitLayer l2(handle);
		Block      b3_1(handle,{1},1);
		Block      b3_2(handle,{1},1);
		DenseLayer l3_1(handle,1);
		DenseLayer l3_2(handle,1);
		Block      b4_1(handle,{1},1);
		Block      b4_2(handle,{1},1);

		//Link
		link(b1,l1,b2);
		link(b2,l2,{b3_1,b3_2});
		link(b3_1,l3_1,b4_1);
		link(b3_2,l3_2,b4_2);

		//Set the filter/bias and input
		b1.data(0)[0]    = 1.0;
		l1.filter()[0]   = 2.0;

		l3_1.filter()[0] = 1.0;
		l3_2.filter()[0] = 4.0;

		b4_1.delta(0)[0] = 1.0;
		b4_2.delta(0)[0] = 1.0;

		//f/b/u
		l1.forward(0);
		l2.forward(0);
		l3_1.forward(0);
		l3_2.forward(0);

		l3_2.backward(0);
		l3_1.backward(0);
		l2.backward(0);
		l1.backward(0);

		l3_2.update(0);
		l3_1.update(0);
		l2.update(0);
		l1.update(0);

		REQUIRE(b4_1.data(0)[0] == 2 );
		REQUIRE(b4_2.data(0)[0] == 8 );
		REQUIRE(b1.delta(0)[0]  == 10);
	}
	// Destroy handle
	cudnnDestroy(handle);
}


TEST_CASE( "PoolingLayer test", "[PoolingLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block        b1(handle,{1},1,2,2);
	PoolingLayer l2(handle);
	Block        b3(handle,{1},1);

	//Set block 1 data
	b1.data(0,0,0,0,0) = 1.0;
	b1.data(0,0,0,0,1) = 2.0;
	b1.data(0,0,0,1,0) = 3.0;
	b1.data(0,0,0,1,1) = 4.0;

	//Set blocks
	link(b1,l2,b3);

	//Forward
	l2.forward(0);

	REQUIRE(b3.data(0)[0] == 4.0);

	//Backward
	l2.backward(0);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "DropoutLayer test", "[DropoutLayer]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Test constructing the Dense layer
	Block        b1(handle,{2,1},10,3.0);
	DropoutLayer d2(handle);
	Block        b3(handle,{2,1},10);

	//Link blocks and layer
	link(b1,d2,b3);

	//Forward value
	d2.forward(0);
	d2.forward(1);

	//Test correctness
	REQUIRE(b3.data(1,0,0,0,0)==1.5);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "Asynchronous Layer Pass test", "[Async]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Require streams for async operations
	cudaStream_t s1;
	cudaStream_t s2;
	check( cudaStreamCreate(&s1) );
	check( cudaStreamCreate(&s2) );

	//Test constructing the Dense layer
	Block      b11(handle,{10},1024);
	DenseLayer d21(handle,4096);
	Block      b31(handle,{10},4096);

	//Test constructing the Dense layer
	Block      b12(handle,{20},1024);
	DenseLayer d22(handle,4096);
	Block      b32(handle,{20},4096);

	//Test set method
	link(b11,d21,b31);
	link(b12,d22,b32);

	//Async Forwards
	for(int i=0; i<100; i++)
	{
		auto future1 = async(launch::async, [&]{return d21.forward(0,s1);} );
		auto future2 = async(launch::async, [&]{return d22.backward(0,s2);} );
		future1.get(); future2.get();
		// d21.forward(0,s1);
		// d22.backward(0,s2);
	}

	//Destroy streams
	check( cudaStreamDestroy(s1) );
	check( cudaStreamDestroy(s2) );

	// Destroy handle
	cudnnDestroy(handle);	
}

TEST_CASE( "Adapter test 1", "[Adapter1]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Build network
	Block      block1(handle,{3},2);
	DenseLayer layer1(handle,1);
	Block      block2(handle,{3},1);
	
	//link together layers
	link(block1,layer1,block2);
	
	//Fill bottom data with all two's
	cudnn_fill(2.0,block1.data(0));

	//Fill top delta with all ones
	cudnn_fill(1.0,block2.delta(0));

	//Make layer adaptor
	Adapter adaptor(make_shared<DenseLayer>(layer1),{0});

	//Forward data
	adaptor.element().forward(0);

	//Backward data
	adaptor.element().backward(0);
	
	//Compute weight derivatives 
	adaptor.element().update(0);

	REQUIRE(block1.data(0)[0] == 2);
	REQUIRE(adaptor.gradients(0)[0] == 2);

	assert(block1.data(0)[0] == 2 and adaptor.gradients(0)[0] == 2);

	//Update weights (Batch Gradient Descent)
	for(int i=0; i<adaptor.size(); i++)
		cudnn_add(-0.01,adaptor.gradients(0),1.0,adaptor.weights(0));
	
	REQUIRE(block1.data(0)[0] == 2);
	REQUIRE(adaptor.gradients(0)[0] == 2);

	// Destroy handle
	cudnnDestroy(handle);
}

TEST_CASE( "Solver test", "[Solver]" )
{
	//Require handle for operations
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Build network
	Block      block1(handle,{1},1024);
	DenseLayer layer1(handle,2048);
	Block      block2(handle,{1},2048);
	
	//link together layers
	link(block1,layer1,block2);
	
	//Fill bottom data with all two's
	cudnn_fill(2.0,block1.data(0));

	//Fill top delta with all ones
	cudnn_fill(1.0,block2.delta(0));

	//Make element adaptor
	Adapter adaptor(make_shared<DenseLayer>(layer1),{0});

	//Make solver
	Solver solver;

	for(int i=0; i<100; i++)
	{
		// Forward data
		adaptor.element().forward(0);
		//Backward data
		adaptor.element().backward(0);
		//Compute weight derivatives 
		adaptor.element().update(0);
		//Solve
		solver.step(adaptor);
	}

	// Destroy handle
	cudnnDestroy(handle);
}


TEST_CASE( "Network test 1", "[Network1]" )
{
	//Require handle for operations	
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Create network
	Network network(handle,{32});
	//Build topology
	auto b1_1 = network.create<Block>(1024);
	auto b1_2 = network.create<Block>(1024);
	auto l1   = network.create<MergeLayer>();
	auto b2   = network.create<Block>(2048);
	auto l2   = network.create<DenseLayer>(4096);
	auto b3   = network.create<Block>(4096);
	auto l3   = network.create<ActivationLayer>();
	auto b4   = network.clone(b3);
	//Create Streams
	auto s    = network.create<Stream>(BGD(-0.007,0.9),vector<Block>{b1_1,b1_2},l1,b2,l2,b3,l3,b4);
	//Set Streams
	network.push(s);
	//Set inputs
	network.inputs(b1_1,b1_2);
	//Set outputs
	network.outputs(b4);

	//Print device memory
	check( cudaDeviceSynchronize() );

	//Forward data
	for(int i=0; i<10; i++)
		network.forward(0);

	//backward data
	for(int i=0; i<10; i++)
		network.backward(0);

	//update data
	for(int i=0; i<10; i++)
		network.update(0);

	//Train
	for(int i=0; i<10; i++)
	{
		network.forward(0);
		network.backward(0);
		network.update(0);
	}

	// Destroy handle
	cudnnDestroy(handle);
}

//Load the MNSIT data set from disk
thrust::device_vector<float> load_mist_labels(string file)
{
	/*READ LABELS*/
		//Open MNSIT data file for reading
		ifstream linf(file+"t10k-labels.idx1-ubyte", ios::binary);
		//Set exception mask
		linf.exceptions(ios::badbit | ios::failbit);
		//Skip 8 bytes
		linf.seekg(8);
		//Build temporary buffer
		array<unsigned char, 1000> buf;
		//Read labels
		linf.read((char*)buf.data(),1000);
		//Cast from unsigned char to float
		array<float, 1000> cbuf;
		copy(buf.begin(),buf.end(),cbuf.begin());
		//Construct device_vector
		thrust::device_vector<float> labels(cbuf.data(),cbuf.data()+1000);
	//Return labels
	return labels;
}

//Load the MNSIT data set from disk
thrust::device_vector<float> load_mist_images(string file)
{
	/*READ IMAGES*/
		//Open MNSIT data file for reading
		ifstream iinf(file+"t10k-images.idx3-ubyte", ios::binary);
		//Set exception mask
		iinf.exceptions(ios::badbit | ios::failbit);
		//Skip 16 bytes
		iinf.seekg(16);
		//Host vector buffer
		thrust::host_vector<float> ibuf(1000*28*28);
		//Variable to temporarily hold value
		char byte;
		//Fill host_vector
		for(int i=0; i<1000; i++)
			for(int r=0; r<28; r++)
				for(int c=0; c<28; c++)
				{
					iinf.read(&byte,1);
					ibuf[i*28*28+c*28+r] = float((unsigned char)byte)/255.0;
				}
		//Construct device_vector from host vector
		thrust::device_vector<float> images(ibuf);
	//Return images
	return images;
}

void print_mnist()
{
	//Load MNSIT data set
	auto labels = load_mist_labels("./data/mnist/");
	auto images = load_mist_images("./data/mnist/");

	for(int x=0; x<28; x++)
	{
		for(int y=0; y<28; y++)
			cout << setprecision(1) << images[x*28+y] << " ";
		cout << endl;
	}
}


TEST_CASE( "Network test 2", "[Network]" )
{
	//Require handle for operations	
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Create network
	auto network = MNIST(handle,{32});

	//Load MNSIT data set
	auto labels = load_mist_labels("./data/mnist/");
	auto images = load_mist_images("./data/mnist/");
	
	//Copy 32 images
	thrust::copy(images.begin(),images.begin()+32*28*28,network.input(0).data(0).begin());

	//Train the network while profiling
	for(int i=0; i<10000; i++)
	{
		//Forward network
		network.forward(0);

		//Set the class label
		set_target_values(labels,network.output(0),0);

		//Backward network
		network.backward(0);
		network.update(0);
	}

	// Destroy handle
	cudnnDestroy(handle);
}