#include "Block.h"

namespace ddnet
{
	//Convolution style constructor
	Block::Block(cudnnHandle_t handle, std::vector<int> n, int c, int h, int w, float val) : data_(handle,n,c,h,w,val), delta_(handle,n,c,h,w,val), n_(n), c_(c), h_(h), w_(w) {}
	
	//Flat style constructor
	Block::Block(cudnnHandle_t handle, std::vector<int> n, int d, float val) : Block(handle,n,1,1,d,val) {}

	//Equality operator
	bool Block::operator==(const Block& rhs) const
	{
		return (data_ == rhs.data_ and delta_ == rhs.delta_);
	}

	//Inequality operator
	bool Block::operator!=(const Block& rhs) const
	{
		return (data_ != rhs.data_ or delta_ != rhs.delta_);
	}

	//Reshape the TensorArray's
	void Block::reshape(std::vector<int> n, int c, int h, int w)
	{
		if(n_.size() != n.size()) throw ddnet_exception(3);
		data_.reshape(n,c,h,w); delta_.reshape(n,c,h,w);
		n_=n; c_=c; h_=h; w_=w;
	}

	//Retun the size of data/delta Tensor i
	int Block::size(int i)
	{
		return data_(i).size();
	}

	//Get one of the dimentions
	std::vector<int> Block::n() const {return n_;}
	int Block::c() const {return c_;}
	int Block::h() const {return h_;}
	int Block::w() const {return w_;}

	//Direct TensorArray interface
	TensorArray& Block::data()  {return data_;}
	TensorArray& Block::delta() {return delta_;}

	//Data thrust interface
	Tensor&                         Block::data(int t)                             {return data_(t);          }
	thrust::device_ptr<float>       Block::data(int t, int n)                      {return data_(t,n);        }
	thrust::device_ptr<float>       Block::data(int t, int n, int c)               {return data_(t,n,c);      }
	thrust::device_ptr<float>       Block::data(int t, int n, int c, int h)        {return data_(t,n,c,h);    }
	thrust::device_reference<float> Block::data(int t, int n, int c, int h, int w) {return data_(t,n,c,h,w);  }


	//Delta thrust interface
	Tensor&                         Block::delta(int t)                             {return delta_(t);         }
	thrust::device_ptr<float>       Block::delta(int t, int n)                      {return delta_(t,n);       }
	thrust::device_ptr<float>       Block::delta(int t, int n, int c)               {return delta_(t,n,c);     }
	thrust::device_ptr<float>       Block::delta(int t, int n, int c, int h)        {return delta_(t,n,c,h);   }
	thrust::device_reference<float> Block::delta(int t, int n, int c, int h, int w) {return delta_(t,n,c,h,w); }

	//Ostream print functionality
	std::ostream& operator<<(std::ostream& os, const Block& block)
	{
		os << "Block(n={";
		for(int i=0; i<block.n_.size()-1; i++)
			os << block.n_[i] << ",";
		os<<block.n_[block.n_.size()-1];
		os << "},c=" << block.c_ << ",h=" << block.h_ << ",w=" << block.w_ << ")";
	    return os;
	}
}