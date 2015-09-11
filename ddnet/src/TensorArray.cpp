#include "TensorArray.h"

namespace ddnet
{
	//Convolution style constuctor
	TensorArray::TensorArray(cudnnHandle_t handle, std::vector<int> n, int c, int h, int w, float val) : n_(n), c_(c), h_(h), w_(w)
	{
		memory_ = std::vector<Tensor>(n.size());
		for(int i=0; i<n.size(); i++)
			memory_[i] = Tensor(handle,n[i],c,h,w,val);
	}

	//Flat style constructor
	TensorArray::TensorArray(cudnnHandle_t handle, std::vector<int> n, int d, float val) : TensorArray(handle,n,1,1,d,val) {}

	//Equality operator
	bool TensorArray::operator==(const TensorArray& rhs) const
	{
		return (memory_ == rhs.memory_);
	}

	//Inequality operator
	bool TensorArray::operator!=(const TensorArray& rhs) const
	{
		return (memory_ != rhs.memory_);
	}

	//Batch reshape the tensors
	void TensorArray::reshape(std::vector<int> n, int c, int h, int w)
	{
		if(n_.size() != n.size()) throw ddnet_exception(3);
		for(int i=0; i<n.size(); i++)
			memory_[i].reshape(n[i],c,h,w);
		n_ = n; c_ = c; h_ = h; w_ = w;
	}

	//Get one of the dimentions
	std::vector<int> TensorArray::n() const 
	{
		return n_;
	}

	int TensorArray::c() const 
	{
		return c_;
	}

	int TensorArray::h() const 
	{
		return h_;
	}

	int TensorArray::w() const 
	{
		return w_;
	}

	//Return the size of Tensor i
	int TensorArray::size(int i)
	{
		return memory_[i].size();
	}

	//Thrust STL Interface
	Tensor& TensorArray::operator()(int t)                                                      
	{
		return memory_[t];
	}

	thrust::device_ptr<float> TensorArray::operator()(int t, int n)                            
	{
		return memory_[t](n);
	}

	thrust::device_ptr<float> TensorArray::operator()(int t, int n, int c)                      
	{
		return memory_[t](n,c);
	}

	thrust::device_ptr<float> TensorArray::operator()(int t, int n, int c, int h)               
	{
		return memory_[t](n,c,h);
	}

	thrust::device_reference<float> TensorArray::operator()(int t, int n, int c, int h, int w)  
	{
		return memory_[t](n,c,h,w);
	}

	//Ostream print functionality
	std::ostream& operator<<(std::ostream& os, const TensorArray& array)
	{
		os << "TensorArray(n={";
		for(int i=0; i<array.n().size()-1; i++)
			os << array.n()[i] << ",";
		os<<array.n()[array.n().size()-1];
		os << "},c=" << array.c() << ",h=" << array.h() << ",w=" << array.w() << ")";
	    return os;
	}
}