// #include "Constraint.h"
// #include <thrust/transform.h>
// #include "thrust/for_each.h"
// #include "thrust/iterator/counting_iterator.h"

// namespace ddnet
// {
// 	//Constructor
// 	Constraint::Constraint(cudnnHandle_t handle, std::vector<std::pair<float,float>> domain) : ActivationLayer(handle,CUDNN_ACTIVATION_SIGMOID), domain_(domain)
// 	{}

// 	//Set output Block
// 	void Constraint::set_out(Block &block)
// 	{
// 		//Get the size of the output
// 		unsigned int size = block.c()*block.h()*block.w();

// 		//Check correctness
// 		if(size != domain_.size()) 
// 			throw "OMFG EXCEPTION!";

// 		//Create the upper and lower device bounds
// 		lower_bound = thrust::device_vector<float>(size);
// 		upper_bound = thrust::device_vector<float>(size);

// 		//Fill the upper and lower device bounds
// 		for(int i=0; i<size; i++)
// 		{
// 			lower_bound[i] = domain_[i].first;
// 			upper_bound[i] = domain_[i].second;
// 		}

// 		//Set the block as normal
// 		ActivationLayer::set_out(block);
// 	}

// 	//Forward Kernel
// 	struct forward_kernel
// 	{
// 		float* lower_bound;
// 		float* upper_bound;
// 		float* activation;
// 		int    n;
// 		forward_kernel(float* lower_bound, float* upper_bound, float* activation, int n) : lower_bound(lower_bound), upper_bound(upper_bound), activation(activation), n(n) {}
// 		__device__ void operator()(const int i)
// 		{
// 			activation[i] = lower_bound[i%n] + (upper_bound[i%n]-lower_bound[i%n])*activation[i];
// 		}
// 	};

// 	//Forward data
// 	Element& Constraint::forward(int i, cudaStream_t stream) 
// 	{
// 		//Normal sigmoid activiation
// 		ActivationLayer::forward(i,stream);

// 		//Counting itterator
// 		thrust::counting_iterator<int> first(0);
// 		thrust::counting_iterator<int> last(output.size(i));

// 		//Extract information
// 		float* l = thrust::raw_pointer_cast(lower_bound.data());
// 		float* u = thrust::raw_pointer_cast(upper_bound.data());
// 		float* a = output.data(i).raw();
// 		int    n = domain_.size();
		
// 		//Apply kernel
// 		thrust::for_each(first, last, forward_kernel(l,u,a,n) );
		
// 		//Return
// 		return *this;
// 	}

// 	//Backward Kernel
// 	struct backward_kernel
// 	{
// 		float* lower_bound;
// 		float* upper_bound;
// 		float* output;
// 		float* output_delta;
// 		float* input;
// 		float* input_delta;
// 		int    n;
// 		backward_kernel(float* lower_bound, float* upper_bound, float* output, float* output_delta, float* input, float* input_delta, int n) : lower_bound(lower_bound), upper_bound(upper_bound), output(output), output_delta(output_delta), input(input), input_delta(input_delta), n(n) {}
// 		__device__ float sigmoid_derivative(float x)
// 		{
// 			float sigm = 1.0/(1+exp(-x));
// 			return (1.0-sigm)*sigm;
// 		}
// 		__device__ void operator()(const int i)
// 		{
// 			input_delta[i] = 2*output_delta[i]*(upper_bound[i%n]-lower_bound[i%n])*sigmoid_derivative(input[i]);
// 		}
// 	};

// 	//Backward data
// 	Element& Constraint::backward(int i, cudaStream_t stream) 
// 	{

// 		//Counting itterator
// 		thrust::counting_iterator<int> first(0);
// 		thrust::counting_iterator<int> last(output.size(i));

// 		//Extract information
// 		float* l  = thrust::raw_pointer_cast(lower_bound.data());
// 		float* u  = thrust::raw_pointer_cast(upper_bound.data());
// 		float* o  = output.data(i).raw();
// 		float* od = output.delta(i).raw();
// 		float* in = input.data(i).raw();
// 		float* id = input.delta(i).raw();
// 		int    n  = domain_.size();
		
// 		//Apply kernel
// 		thrust::for_each(first, last, backward_kernel(l,u,o,od,in,id,n) );
		
// 		//Return
// 		return *this;
// 	}
// }
