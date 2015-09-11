#include "Convolution.h"

namespace ddnet
{
	Filter::Filter(cudnnHandle_t handle, int k, int c, int h, int w, float var) : Tensor(handle,k,c,h,w) 
	{
		//Create filter descriptor
		check( cudnnCreateFilterDescriptor(&fDescriptor_) );
		check( cudnnSetFilter4dDescriptor(fDescriptor_,CUDNN_DATA_FLOAT,n_,c_,h_,w_) );
		curand_rand_normal(0.0,var,*this);
	}

	//Copy Constructor (Shallow)
	Filter::Filter(const Filter& s) : Tensor(s)
	{
		//Copy source descriptor if it exists
		if(s.descriptor_)
		{
			check( cudnnCreateFilterDescriptor(&fDescriptor_) );
			check( cudnnSetFilter4dDescriptor(fDescriptor_,CUDNN_DATA_FLOAT, s.n_, s.c_, s.h_, s.w_) );
		}
	}

	//Copy Assignment operator (Shallow)
	Filter& Filter::operator=(const Filter &s)
	{
		//Call Tensor operator first
		Tensor::operator=(s);
		//Copy source descriptor if it exists
		if(s.descriptor_)
		{
			check( cudnnCreateFilterDescriptor(&fDescriptor_) );
			check( cudnnSetFilter4dDescriptor(fDescriptor_,CUDNN_DATA_FLOAT, s.n_, s.c_, s.h_, s.w_) );
		}
		return *this;
	}

	//Move Assignment operator
	Filter& Filter::operator=(Filter &&s)
	{
		//Call Tensor operator first
		Tensor::operator=(s);
		//Copy source descriptor if it exists
		if(s.descriptor_)
		{
			check( cudnnCreateFilterDescriptor(&fDescriptor_) );
			check( cudnnSetFilter4dDescriptor(fDescriptor_,CUDNN_DATA_FLOAT, s.n_, s.c_, s.h_, s.w_) );
		}
		return *this;
	}

	//Change the descriptor
	void Filter::reshape(int k, int c, int h, int w)
	{
		Tensor::reshape(k,c,h,w);
		//Create new descriptor
		check( cudnnDestroyFilterDescriptor(fDescriptor_) );
		check( cudnnCreateFilterDescriptor(&fDescriptor_) );
		check( cudnnSetFilter4dDescriptor(fDescriptor_,CUDNN_DATA_FLOAT,k,c,h,w) );
	}

	cudnnFilterDescriptor_t Filter::fDescriptor() {   return fDescriptor_;   }

	//Thrust STL Interface
	thrust::device_ptr<float> Filter::begin() {return thrust::device_ptr<float>(data_.get());}
	thrust::device_ptr<float> Filter::end()   {return thrust::device_ptr<float>(data_.get()) + n_*c_*h_*w_;}

	thrust::device_reference<float> Filter::operator[](int i)                      { return thrust::device_ptr<float>(data_.get())[i]; }
	thrust::device_ptr<float>       Filter::operator()(int k)                      { return thrust::device_ptr<float>(data_.get()+k*c_*h_*w_); }
	thrust::device_ptr<float>       Filter::operator()(int k, int c)               { return thrust::device_ptr<float>(data_.get()+k*c_*h_*w_+c*h_*w_); }
	thrust::device_ptr<float>       Filter::operator()(int k, int c, int h)        { return thrust::device_ptr<float>(data_.get()+k*c_*h_*w_+c*h_*w_+h*w_); }
	thrust::device_reference<float> Filter::operator()(int k, int c, int h, int w) { return thrust::device_ptr<float>(data_.get())[k*c_*h_*w_+c*h_*w_+h*w_+w]; }

	//Destroy descriptor
	Filter::~Filter()
	{
		if(fDescriptor_!=NULL)
			check( cudnnDestroyFilterDescriptor(fDescriptor_) );
	}

	//Printing functionality
	std::ostream& operator<<(std::ostream& os, const Filter& filter)
	{   os << "Filter(k=" << filter.n() << ",c=" << filter.c() << ",h=" << filter.h() << ",w=" << filter.w() << ")"; return os;   }

	//Constructors
	Algorithm::Algorithm(cudnnHandle_t handle, Tensor &input,  Filter &filter, cudnnConvolutionDescriptor_t convDescriptor , Tensor &output, cudnnConvolutionFwdPreference_t pref)
	{
		//Get the algorithm type
		check( cudnnGetConvolutionForwardAlgorithm(handle,input.descriptor(),filter.fDescriptor(),convDescriptor,output.descriptor(),pref,0,&type_) );
		//Get the size of the workspace
		check( cudnnGetConvolutionForwardWorkspaceSize(handle,input.descriptor(),filter.fDescriptor(),convDescriptor,output.descriptor(),type_,&size_) );
		//Allocate workspace memory
		if(size_!=0)
		{
			float* ptr;
			check( cudaMalloc(&ptr,size_) );
			workspace_ = std::shared_ptr<float>(ptr,cudaFree);
		}
		else
			workspace_ = NULL;
	}

	//Algorithm size in bytes
	size_t Algorithm::size() const 
	{
		return size_;
	}
	
	//Type of algorithm being used
	cudnnConvolutionFwdAlgo_t Algorithm::type() const 
	{
		return type_;
	}
	
	//Workspace pointer
	float* Algorithm::workspace() const 
	{
		return workspace_.get();
	}

	//Return the convolution descriptor
	cudnnConvolutionDescriptor_t Convolution::descriptor() 
	{
		return descriptor_.get();
	}

	//Constructor
	Convolution::Convolution(cudnnHandle_t handle, int f, int fh, int fw, int vs, int hs, float f_var, float b_val, int hz, int wz) : Layer(handle), f_(f), fh_(fh), fw_(fw), f_var_(f_var), b_val_(b_val), vs_(vs), hs_(hs), hz_(hz), wz_(wz) 
	{}

	//Set input Block
	void Convolution::set_in(Block &block)
	{
		//Set the input block
		input = block;

		//Set filter and bias
		filter_       = Filter(handle,f_,input.c(),fh_,fw_,f_var_);
		bias_         = Tensor(handle,1,f_,1,1,b_val_);

		//Set Filter and Bias gradients
		filter_delta_ = Filter(handle,f_,input.c(),fh_,fw_);
		bias_delta_   = Tensor(handle,1,f_,1,1);

		//Create Convolution Descriptor as a shared_ptr
		cudnnConvolutionDescriptor_t dptr;
		check( cudnnCreateConvolutionDescriptor(&dptr) );
		check( cudnnSetConvolution2dDescriptor(dptr,hz_,wz_,vs_,hs_,1,1,CUDNN_CONVOLUTION) );
		descriptor_ = std::shared_ptr<cudnnConvolutionStruct>(dptr,cudnnDestroyConvolutionDescriptor);
	}

	//Set output Block
	void Convolution::set_out(Block &block)
	{
		//Set output block
		output = block;
		//Check concistency
		if( input.n() != output.n() ) throw ddnet_exception(0);
		for(int i=0; i<input.n().size(); i++)
		{
			int n; int c; int h; int w;
			cudnnGetConvolution2dForwardOutputDim(descriptor(),input.data(i).descriptor(),filter_.fDescriptor(),&n,&c,&h,&w);
			if(output.n()[i]!=n or output.c() != c or output.h() != h or output.w() != w) throw ddnet_exception(0);
		}
		//Set the algorithm_s
		algorithm_ = std::vector<Algorithm>( input.n().size() );
		for(int i=0; i<algorithm_.size(); i++)
			algorithm_[i] = Algorithm(handle,input.data(i),filter_,descriptor(),output.data(i));
	}

	//Get filter
	Filter& Convolution::filter()
	{
		return filter_;
	}

	//Get bias
	Tensor& Convolution::bias()
	{
		return bias_; 
	}
	
	//Get filter delta
	Filter& Convolution::filter_delta()
	{
		return filter_delta_;
	}

	//Get bias delta
	Tensor& Convolution::bias_delta()
	{
		return bias_delta_;
	}

	//Apply convolution
	Element& Convolution::forward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Forward data
		float alpha = 1.0; float beta = 0.0;
		check( cudnnConvolutionForward(handle,&alpha,
			input.data(i).descriptor(),input.data(i).raw(),
			filter_.fDescriptor(),filter_.raw(),
			descriptor(),
			algorithm_[i].type(),algorithm_[i].workspace(),algorithm_[i].size(),
			&beta,
			output.data(i).descriptor(),output.data(i).raw()) );
		//Add bias tensor
		cudnn_add(1.0,bias_,1.0,output.data(i),CUDNN_ADD_SAME_C);
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	};

	//Compute the data derivative
	Element& Convolution::backward(int i, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Multipliers
		float alpha = 1.0; float beta = 0.0;
		//Compute DL/Dx
		check( cudnnConvolutionBackwardData(handle,&alpha,
			filter_.fDescriptor(),filter_.raw(),
			output.delta(i).descriptor(),output.delta(i).raw(),
			descriptor(),
			&beta,
			input.delta(i).descriptor(),input.delta(i).raw()) );
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	};

	//Compute the filter derivative
	Element& Convolution::update(int i, bool accumulate, cudaStream_t stream)
	{
		//Set the stream
		check( cudnnSetStream(handle,stream) );
		//Multipliers 
		float alpha = 1.0/input.n()[i]; //(Average over mini-batch)
		float beta = accumulate;
		//Compute DL/Dw 
		check( cudnnConvolutionBackwardFilter(handle,&alpha,
			input.data(i).descriptor(),input.data(i).raw(),
			output.delta(i).descriptor(),output.delta(i).raw(),
			descriptor(),&beta,
			filter_delta_.fDescriptor(),filter_delta_.raw()) );
		//Compute DL/Db
		alpha = 1.0;
		check( cudnnConvolutionBackwardBias(handle,&alpha,
			output.delta(i).descriptor(),output.delta(i).raw(),
			&beta,
			bias_delta_.descriptor(),bias_delta_.raw()) );
		//Set back to NULL stream
		check( cudnnSetStream(handle,NULL) );
		return *this;
	};

	//Return filter and bias
	std::vector<Tensor> Convolution::weights()
	{
		return {filter_,bias_};
	}

	//Return filter and bias gradients
	std::vector<Tensor> Convolution::gradients() 
	{
		return {filter_delta_,bias_delta_};
	}

}