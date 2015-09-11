#pragma once
#include "Layer.h"

namespace ddnet
{
	/*
		Filter class manages filter memory it's descriptor. It takses Tensor as it's base clas, this allows us to easily work with the filter as if it was a Tensor.
			k: Filter input feature maps
			c: Filter output feature maps
			h: Filter height
			w: Filter width
		This class uses the functionality from the Tensor class for manipulation
	*/
	class Filter : public Tensor
	{
	protected:
		cudnnFilterDescriptor_t  fDescriptor_ = 0;
	public:
		//Default constructor
		Filter() = default;
		//Constructor
		Filter(cudnnHandle_t handle, int k, int c, int h, int w, float var = 0.01);
		//Copy Constructor (Shallow)
		Filter(const Filter& s);
		//Copy Assignment operator (Shallow)
		Filter& operator=(const Filter &s);
		//Move Assignment operator
		Filter& operator=(Filter &&s);
		//Change the descriptor
		void reshape(int k, int c, int h, int w);
		//Get filter descriptor
		cudnnFilterDescriptor_t fDescriptor();
		//Thrust STL Interface
		thrust::device_ptr<float> begin();
		thrust::device_ptr<float> end();
		thrust::device_reference<float> operator[](int i);
		thrust::device_ptr<float> operator()(int k);
		thrust::device_ptr<float> operator()(int k, int c);
		thrust::device_ptr<float> operator()(int k, int c, int h);
		thrust::device_reference<float> operator()(int k, int c, int h, int w);
		//Destructor
		~Filter();
		//Printing functionality
		friend std::ostream& operator<<(std::ostream& os, const Filter& filter);
	};

	/*
		The Algorithm class encapsulates all the algorithm data, together with the workspace. This saves us trouble keeping track and initializing the algorithm. 
		Because Algorithm tracks device memory, copies are shallow and workspace memory is only created when the main constructer is called.
	*/
	class Algorithm
	{
	protected:
		cudnnHandle_t handle = 0;
		std::shared_ptr<float> workspace_ = 0;
		cudnnConvolutionFwdAlgo_t type_;
		size_t size_ = 0;
	public:
		//Default Constructor
		Algorithm() = default;
		//Constructor
		Algorithm(cudnnHandle_t handle, Tensor &input,  Filter &filter, cudnnConvolutionDescriptor_t convDescriptor , Tensor &output, cudnnConvolutionFwdPreference_t pref = CUDNN_CONVOLUTION_FWD_PREFER_FASTEST);
		//Algorithm size in bytes
		size_t size() const;
		//Type of algorithm being used
		cudnnConvolutionFwdAlgo_t type() const;
		//Workspace pointer
		float* workspace() const;
	};

	/*
		The convolution layer class sets all the nessesary cudnn descriptors and creates the Filter filter and Tensor bias. It is the largest class and
		takes most of the compute power.
	*/
	class Convolution : public Layer
	{
	protected:
		std::vector<Algorithm> algorithm_;
		std::shared_ptr<cudnnConvolutionStruct> descriptor_ = 0;
		
		int f_       = 0;  //Number of filters
		int fh_      = 0;  //Filter width
		int fw_      = 0;  //Filter height
		int vs_      = 0;  //Verticle stride
		int hs_      = 0;  //Horizontal stride
		int hz_      = 0;  //Padding height
		int wz_      = 0;  //Padding width
		float f_var_ = 0;  //weight initialization variance
		float b_val_ = 0;  // bias initial value

		Filter filter_;
		Filter filter_delta_;
		Tensor bias_;
		Tensor bias_delta_;

		//Return the convolution descriptor
		cudnnConvolutionDescriptor_t descriptor();

	public:
		//Default Constructor
		Convolution() = default;
		//Constructor
		Convolution(cudnnHandle_t handle, int f, int fh, int fw, int vs = 1, int hs = 1, float f_var = 0.01, float b_val = 0.02, int hz = 0, int wz = 0);
		//Set input Block
		virtual void set_in(Block &block) override;
		//Set output Block
		virtual void set_out(Block &block) override;
		//Get filter
		Filter& filter();
		//Get bias
		Tensor& bias();
		//Get filter delta
		Filter& filter_delta();
		//Get bias delta
		Tensor& bias_delta();
		//Apply convolution
		virtual Element& forward(int i, cudaStream_t stream = NULL) override;
		//Compute the data derivative
		virtual Element& backward(int i, cudaStream_t stream = NULL) override;
		//Compute the filter derivative
		virtual Element& update(int i, bool accumulate = false, cudaStream_t stream = NULL) override;
		//Return filter and bias
		virtual std::vector<Tensor> weights() override;
		//Return filter and bias gradients
		virtual std::vector<Tensor> gradients() override;
	};
}