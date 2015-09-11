#include "Tensor.h"

namespace ddnet
{
	//Create a convolution style tensor n=#images, c=#channels, h=heightm w=width
	Tensor::Tensor(cudnnHandle_t handle, int n, int c, int h, int w, float val) : handle_(handle), n_(n), c_(c), h_(h), w_(w)
	{
		//Create the descriptor
		check( cudnnCreateTensorDescriptor(&descriptor_) );
		check( cudnnSetTensor4dDescriptor(descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_, c_, h_, w_) );
		//Allocate Device Memory (shared_ptr to manage shallow copies)
		float* ptr;
		check( cudaMalloc((void**)&ptr, sizeof(float)*n_*c_*h_*w_) );
		data_ = std::shared_ptr<float>(ptr,cudaFree);
		//Fill the Tensor to make sure we start with sane values
		cudnn_fill(val,*this);
	}
	
	//Create a flat style tensor int=#images, d=dimentions(#neurons)
	Tensor::Tensor(cudnnHandle_t handle_, int n, int d, float val) : Tensor(handle_,n,1,1,d,val) {}


	//Copy Constructor (Shallow)
	Tensor::Tensor(const Tensor &s) : handle_(s.handle_), data_(s.data_), descriptor_(s.descriptor_), n_(s.n_), c_(s.c_), h_(s.h_), w_(s.w_)
	{
		//Copy source descriptor if it exists
		if(s.descriptor_)
		{
			check( cudnnCreateTensorDescriptor(&descriptor_) );
			check( cudnnSetTensor4dDescriptor(descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s.n_, s.c_, s.h_, s.w_) );
		}
	}

	//Move Constructor (Shallow)
	Tensor::Tensor(Tensor &&s) : handle_(s.handle_), data_(s.data_), descriptor_(s.descriptor_), n_(s.n_), c_(s.c_), h_(s.h_), w_(s.w_)
	{
		//Set source descriptor_ to NULL
		s.descriptor_ = 0;
	}

	//Assignment operator (Shallow)
	Tensor& Tensor::operator=(const Tensor &s)
	{
		//Copy member variables
		descriptor_ = s.descriptor_; handle_ = s.handle_; data_ = s.data_; n_ = s.n_; c_ = s.c_; h_ = s.h_; w_ = s.w_;
		//Copy source descriptor if it exists
		if(s.descriptor_)
		{
			check( cudnnCreateTensorDescriptor(&descriptor_) );
			check( cudnnSetTensor4dDescriptor(descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s.n_, s.c_, s.h_, s.w_) );
		}
		return *this;
	}

	//Move Assignment operator
	Tensor& Tensor::operator=(Tensor &&s)
	{
		//Copy member variables
		descriptor_ = s.descriptor_; handle_ = s.handle_; data_ = s.data_; n_ = s.n_; c_ = s.c_; h_ = s.h_; w_ = s.w_;
		//Set source to NULL
		s.descriptor_ = 0;
		return *this;
	}

	//Equality operator
	bool Tensor::operator==(const Tensor& rhs) const
	{
		return (data_ == rhs.data_);
	}

	//Inequality operator
	bool Tensor::operator!=(const Tensor& rhs) const 
	{
		return (data_ != rhs.data_);
	}

	//Return Tensor size
	int Tensor::size()
	{
		return n_*c_*h_*w_;
	}

	//Change the descriptor
	void Tensor::reshape(int n, int c, int h, int w)
	{
		if( n_*c_*h_*w_ != n*c*h*w ) throw ddnet_exception(3);
		//Create new descriptor
		check( cudnnDestroyTensorDescriptor(descriptor_) );
		check( cudnnCreateTensorDescriptor(&descriptor_) );
		check( cudnnSetTensor4dDescriptor(descriptor_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w) );
		n_=n; c_=c; h_=h; w_=w;
	}

	//Get the descriptor
	cudnnTensorDescriptor_t Tensor::descriptor() const 
	{
		return descriptor_;
	}

	//Get one of the dimentions
	int Tensor::n() const {return n_;}
	int Tensor::c() const {return c_;}
	int Tensor::h() const {return h_;}
	int Tensor::w() const {return w_;}

	//Thrust STL Interface
	thrust::device_ptr<float>       Tensor::begin()
	{   return thrust::device_ptr<float>(data_.get());  }

	thrust::device_ptr<float>       Tensor::end()
	{   return thrust::device_ptr<float>(data_.get()) + n_*c_*h_*w_;  }

	thrust::device_reference<float> Tensor::operator[](int i)
	{   return thrust::device_ptr<float>(data_.get())[i];   }

	thrust::device_ptr<float>       Tensor::operator()(int n)
	{   return thrust::device_ptr<float>(data_.get()+n*c_*h_*w_);   }

	thrust::device_ptr<float>       Tensor::operator()(int n, int c)
	{   return thrust::device_ptr<float>(data_.get()+n*c_*h_*w_+c*h_*w_);   }

	thrust::device_ptr<float>       Tensor::operator()(int n, int c, int h)
	{   return thrust::device_ptr<float>(data_.get()+n*c_*h_*w_+c*h_*w_+h*w_);   }

	thrust::device_reference<float> Tensor::operator()(int n, int c, int h, int w)
	{   return thrust::device_ptr<float>(data_.get())[n*c_*h_*w_+c*h_*w_+h*w_+w];   }


	//Raw Pointer Interface
	float* Tensor::raw() const 
	{
		return data_.get();
	}
	float* Tensor::raw(int n) const 
	{
		return data_.get()+(n*c_*h_*w_);
	}
	float* Tensor::raw(int n, int c) const 
	{
		return data_.get()+(n*c_*h_*w_)+(c*h_*w_);
	}

	//Tensor destructor
	Tensor::~Tensor()
	{
		if(descriptor_!=0)
			check( cudnnDestroyTensorDescriptor(descriptor_) );
	}

	//Duplicate the Tensor
	Tensor duplicate(Tensor &t, float val)
	{
		return Tensor(t.handle_,t.n_,t.c_,t.h_,t.w_,val);
	}

	//Printing functionality
	std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
	{   os << "Tensor(n=" << tensor.n() << ",c=" << tensor.c() << ",h=" << tensor.h() << ",w=" << tensor.w() << ")"; return os;   }

	void copy(Tensor &src, Tensor &dst, float alpha, float beta)
		{   check( cudnnTransformTensor(src.handle_,&alpha,src.descriptor(),src.raw(),&beta,dst.descriptor(),dst.raw()) );   }

	//Transform the src and copy it into the dst (dst is shifted)
	void transform_src(Tensor &src, Tensor &dst, int shift, int ns, int cs, int hs, int ws, float alpha, float beta)
	{
		//Get the original stride etc
		cudnnDataType_t type; int n; int c; int h; int w; int stride_n; int stride_c; int stride_h; int stride_w;
		check( cudnnGetTensor4dDescriptor(src.descriptor(),&type,&n,&c,&h,&w,&stride_n,&stride_c,&stride_h,&stride_w) );
		//Create temporary strided descriptor
		cudnnTensorDescriptor_t strided;
		check( cudnnCreateTensorDescriptor(&strided) );
		check( cudnnSetTensor4dDescriptorEx(strided,CUDNN_DATA_FLOAT,n,c,h,w,stride_n+ns,stride_c+cs,stride_h+hs,stride_w+ws) );
		//Transfrom tensor
		check( cudnnTransformTensor(src.handle_,&alpha,src.descriptor(),src.raw(),&beta,strided,dst.raw()+shift) );
		check( cudnnDestroyTensorDescriptor(strided) );
	}

	//Transform the dst and copy src into dst (src is shifted)
	void transform_dst(Tensor &src, Tensor &dst, int shift, int ns, int cs, int hs, int ws, float alpha, float beta)
	{
		//Get the original stride etc
		cudnnDataType_t type; int n; int c; int h; int w; int stride_n; int stride_c; int stride_h; int stride_w;
		check( cudnnGetTensor4dDescriptor(dst.descriptor(),&type,&n,&c,&h,&w,&stride_n,&stride_c,&stride_h,&stride_w) );
		//Create temporary strided descriptor
		cudnnTensorDescriptor_t strided;
		check( cudnnCreateTensorDescriptor(&strided) );
		check( cudnnSetTensor4dDescriptorEx(strided,CUDNN_DATA_FLOAT,n,c,h,w,stride_n+ns,stride_c+cs,stride_h+hs,stride_w+ws) );
		//Transfrom tensor
		check( cudnnTransformTensor(src.handle_,&alpha,strided,src.raw()+shift,&beta,dst.descriptor(),dst.raw()) );
		check( cudnnDestroyTensorDescriptor(strided) );
	}

	/*Function interface to other CUDA libraries*/
	void cudnn_transform(Tensor &src, Tensor &dst, int ns, int cs, int hs, int ws, float alpha, float beta)
	{
		//Create temporary strided descriptor
		cudnnTensorDescriptor_t strided;
		check( cudnnCreateTensorDescriptor(&strided) );
		check( cudnnSetTensor4dDescriptorEx(strided,CUDNN_DATA_FLOAT,src.n(),src.c(),src.h(),src.w(),ns,cs,hs,ws) );
		//Transfrom tensor
		check( cudnnTransformTensor(src.handle_,&alpha,src.descriptor(),src.raw(),&beta,strided,dst.raw()) );
		check( cudnnDestroyTensorDescriptor(strided) );
	}

	void cudnn_fill(float value, Tensor &tensor)
	{   check( cudnnSetTensor(tensor.handle_,tensor.descriptor(),tensor.raw(),&value) );   }

	void cudnn_add(float alpha, Tensor &src, float beta, Tensor &dst, cudnnAddMode_t mode)
	{   check( cudnnAddTensor(dst.handle_,mode,&alpha,src.descriptor(),src.raw(),&beta,dst.descriptor(),dst.raw()) );   }

	void cudnn_scale(float alpha, Tensor &tensor)
	{   check( cudnnScaleTensor(tensor.handle_,tensor.descriptor(),tensor.raw(),&alpha) );   }

	void curand_rand_uniform(Tensor &tensor, cudaStream_t stream)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_MT19937);
		curandSetStream(generator,stream);
		curandSetPseudoRandomGeneratorSeed(generator,(unsigned)time(0));
		curandGenerateUniform(generator, tensor.raw(), tensor.n()*tensor.c()*tensor.h()*tensor.w());
		curandDestroyGenerator(generator);
	}

	void curand_rand_normal(float mean, float std, Tensor &tensor, cudaStream_t stream)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_MTGP32);
		curandSetStream(generator,stream);
		curandSetPseudoRandomGeneratorSeed(generator,(unsigned)time(0));
		curandGenerateNormal(generator, tensor.raw(), tensor.n()*tensor.c()*tensor.h()*tensor.w(), mean, std);
		curandDestroyGenerator(generator);
	}

	void curand_rand_log_normal(float mean, float std, Tensor &tensor, cudaStream_t stream)
	{
		curandGenerator_t generator;
		curandCreateGenerator(&generator,CURAND_RNG_PSEUDO_MTGP32);
		curandSetStream(generator,stream);
		curandSetPseudoRandomGeneratorSeed(generator,(unsigned)time(0));
		curandGenerateLogNormal(generator, tensor.raw(), tensor.n()*tensor.c()*tensor.h()*tensor.w(), mean, std);
		curandDestroyGenerator(generator);
	}
}