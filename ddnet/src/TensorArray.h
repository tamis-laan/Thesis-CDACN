#pragma once
#include "Tensor.h"
#include "Util.h"
#include <vector>

namespace ddnet
{
	/*
		The TensorArray class keep tracks of an array of tensors. Each tensor has the same c h w. The difference is in the n variables. Namely one array may concist of more
		images n than another array. Using the TensorArray class we can track multiple image batches.
			n: A vector specifying the number of images per tensor
			c: Number of channels or filters
			h: Images height
			w: Images width
	*/
	class TensorArray final
	{
	protected:
		std::vector<int> n_; 
		int c_ = 0; 
		int h_ = 0; 
		int w_ = 0;
		std::vector<Tensor> memory_;
	public:
		//Default empty TensorArray constructor
		TensorArray() = default;
		//Convolution style constuctor
		TensorArray(cudnnHandle_t handle, std::vector<int> n, int c, int h, int w, float val = 0.0);
		//Flat style constructor
		TensorArray(cudnnHandle_t handle, std::vector<int> n, int d, float val = 0.0);
		//Equality operator
		bool operator==(const TensorArray& rhs) const;
		//Inequality operator
		bool operator != (const TensorArray& rhs) const;
		//Batch reshape the tensors
		void reshape(std::vector<int> n, int c, int h, int w);
		//Get one of the dimentions
		std::vector<int> n() const;
		int c() const;
		int h() const;
		int w() const;
		//Return the size of Tensor i
		int size(int i);
		//Thrust STL Interface
		Tensor& operator()(int t);
		thrust::device_ptr<float> operator()(int t, int n);
		thrust::device_ptr<float> operator()(int t, int n, int c);
		thrust::device_ptr<float> operator()(int t, int n, int c, int h);
		thrust::device_reference<float> operator()(int t, int n, int c, int h, int w);
		//Ostream print functionality
		friend std::ostream& operator<<(std::ostream& os, const TensorArray& array);
	};
}