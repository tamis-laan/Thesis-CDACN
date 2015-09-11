#pragma once
#include "Util.h"
#include <memory>
#include <thrust/device_ptr.h>

namespace ddnet
{
	/*
	The Tensor class manages memory and it's descriptor. The device memory associated with a Tensor object is only allocated when the constructor is called.
	The assignment and copy operators simply pass on a shared pointer making a effective shallow copy of the Tensor. This means we can have multiple Tensors
	pointing to the same device memories. However each tensor can have it's own descriptor, n, c, h, w. These variables can be altered per Tensor by calling
	the reshape method.
		n: Number of images
		c: Number of channels or filters
		h: Images height
		w: Image width
	Data Layout: NCHW
	*/
	class Tensor
	{
	protected:
		cudnnHandle_t handle_               = 0;
		std::shared_ptr<float> data_        = 0;
		cudnnTensorDescriptor_t descriptor_ = 0;
		int n_ = 0; 
		int c_ = 0; 
		int h_ = 0; 
		int w_ = 0;
	public:
		//Default empty constructor
		Tensor() = default;
		//Create a convolution style tensor n=#images, c=#channels, h=heightm w=width
		Tensor(cudnnHandle_t handle, int n, int c, int h, int w, float val = 0); 
		//Create a flat style tensor int=#images, d=dimentions(#neurons)
		Tensor(cudnnHandle_t handle_, int n, int d, float val = 0.0);
		//Copy Constructor (Shallow)
		Tensor(const Tensor &s);
		//Move Constructor (Shallow)
		Tensor(Tensor &&s);
		//Assignment operator (Shallow)
		Tensor& operator=(const Tensor &s);
		//Move Assignment operator
		Tensor& operator=(Tensor &&s);
		//Equality operator
		bool operator==(const Tensor& rhs) const;
		//Inequality operator
		bool operator!=(const Tensor& rhs) const;
		//Return Tensor size
		int size();
		//Change the descriptor
		void reshape(int n, int c, int h, int w);
		//Get the descriptor
		cudnnTensorDescriptor_t descriptor() const;
		//Get one of the dimentions
		int n() const;
		int c() const;
		int h() const;
		int w() const;
		//Thrust STL Interface
		thrust::device_ptr<float> begin();
		thrust::device_ptr<float> end();
		//Interface
		thrust::device_reference<float> operator[](int i);
		thrust::device_ptr<float> operator()(int n);
		thrust::device_ptr<float> operator()(int n, int c);
		thrust::device_ptr<float> operator()(int n, int c, int h);
		thrust::device_reference<float> operator()(int n, int c, int h, int w);
		//Raw Pointer Interface
		float* raw() const;
		float* raw(int n) const;
		float* raw(int n, int c) const;
		//Tensor destructor
		~Tensor();
		//Duplicate the Tensor
		friend Tensor duplicate(Tensor &t, float val = 0);
		//Printing functionality
		friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
		friend void copy(Tensor &src, Tensor &dst, float alpha = 1.0, float beta = 0.0);
		//Transform the src and copy it into the dst (dst is shifted)
		friend void transform_src(Tensor &src, Tensor &dst, int shift, int ns = 0, int cs = 0, int hs = 0, int ws = 0, float alpha=1.0, float beta=0.0);
		//Transform the dst and copy src into dst (src is shifted)
		friend void transform_dst(Tensor &src, Tensor &dst, int shift, int ns = 0, int cs = 0, int hs = 0, int ws = 0, float alpha=1.0, float beta=0.0);
		//Transform tensor into another tensor
		friend void cudnn_transform(Tensor &src, Tensor &dst, int ns = 0, int cs = 0, int hs = 0, int ws = 0, float alpha=1.0, float beta=1.0);
		//Fill tensor with value
		friend void cudnn_fill(float value, Tensor &tensor);
		//Add two tensors element wise
		friend void cudnn_add(float alpha, Tensor &src, float beta, Tensor &dst, cudnnAddMode_t mode = CUDNN_ADD_FULL_TENSOR);
		//Scale the tensor data with a value
		friend void cudnn_scale(float alpha, Tensor &tensor);
		//Fill tensor with uniform random data
		friend void curand_rand_uniform(Tensor &tensor, cudaStream_t stream = NULL);
		//Fill tensor with normal random data
		friend void curand_rand_normal(float mean, float std, Tensor &tensor, cudaStream_t stream = NULL);
		//Fill tensor with curand_rand_log_normal normal random data
		friend void curand_rand_log_normal(float mean, float std, Tensor &tensor, cudaStream_t stream = NULL);
	};
}