#pragma once
#include "TensorArray.h"

namespace ddnet
{
	/*
		A block manages two TensorArray's, one is called data and carries around all the information produced bij moving data forward throught the network.
		The second TensorArray keeps track of the delta's (derivative with respect to the loss).
	*/
	class Block final
	{
	protected:
		//Holds data from forward
		TensorArray data_;
		//Holds data from backward
		TensorArray delta_;
		std::vector<int> n_; 
		int c_ = 0; 
		int h_ = 0; 
		int w_ = 0;
	public:
		//Default empty constructor
		Block() = default;
		//Convolution style constructor
		Block(cudnnHandle_t handle, std::vector<int> n, int c, int h, int w, float val = 0.0);
		//Flat style constructor
		Block(cudnnHandle_t handle, std::vector<int> n, int d, float val = 0.0);
		//Equality operator
		bool operator == (const Block& rhs) const;
		//Inequality operator
		bool operator != (const Block& rhs) const;
		//Reshape the TensorArray's
		void reshape(std::vector<int> n, int c, int h, int w);
		//Retun the size of data/delta Tensor i
		int size(int i);
		//Get one of the dimentions
		std::vector<int> n() const;
		int c() const;
		int h() const;
		int w() const;
		//Direct TensorArray interface
		TensorArray& data();
		TensorArray& delta();
		//Data thrust interface
		Tensor& data(int t);
		thrust::device_ptr<float> data(int t, int n);
		thrust::device_ptr<float> data(int t, int n, int c);
		thrust::device_ptr<float> data(int t, int n, int c, int h);
		thrust::device_reference<float> data(int t, int n, int c, int h, int w);
		//Delta thrust interface
		Tensor& delta(int t);
		thrust::device_ptr<float> delta(int t, int n);
		thrust::device_ptr<float> delta(int t, int n, int c);
		thrust::device_ptr<float> delta(int t, int n, int c, int h);
		thrust::device_reference<float> delta(int t, int n, int c, int h, int w);
		//Ostream print functionality
		friend std::ostream& operator<<(std::ostream& os, const Block& block);
		//Convinient function to set labels (NOTE: This not efficient!! Also must be able to use stream!!)
		template<typename T> friend void set_target_values(T &labels, Block &block, int i)
		{
			thrust::copy(labels.begin(), labels.begin() + block.size(i), block.delta(i).begin());
			cudnn_add(-1.0,block.data(i),1.0,block.delta(i));
		}
	};
}