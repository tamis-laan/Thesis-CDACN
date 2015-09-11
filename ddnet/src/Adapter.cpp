#include "Adapter.h"
using namespace ddnet;

Adapter::Adapter(std::shared_ptr<Element> element, std::vector<float> buffer_values) : element_(element)
{
	//Make sure the element has been set accordingly 
	if(not element->guard()) throw ddnet_exception(6);
	//Save the element weigts
	weights_ = element->weights();
	//save the elements gradients
	gradients_ = element->gradients();
	//Build up the buffers
	buffers_ = std::vector<std::vector<Tensor>>(buffer_values.size());
	for(auto &buf : buffers_)
	{
		buf = std::vector<Tensor>( gradients_.size() );
		for(int i=0; i<buf.size(); i++)
			buf[i] = duplicate(gradients_[i],buffer_values[i]);
	}
}

//Return the number of weights
int Adapter::size() {return weights_.size();}

//Return elements
Element& Adapter::element() {return *element_;}
	
//Return parameters
Tensor& Adapter::weights(int i) {return weights_[i];}

//Return gradients
Tensor& Adapter::gradients(int i) {return gradients_[i];}

//Return buffers
std::vector<Tensor>& Adapter::buffers(int i) {return buffers_[i];}

//Return buffers
Tensor& Adapter::buffers(int i, int j) {return buffers_[i][j];}

//Transfer weights to other adapter
void Adapter::transfer(Adapter &a)
{
	//Check equal number of weight Tensors
	if(weights_.size()!=a.weights_.size()) throw ddnet_exception(10);
	for(int i=0; i<weights_.size(); i++)
	{
		//Check equal sized tensors
		if(weights_[i].size() != a.weights_[i].size()) throw ddnet_exception(10);
		//Transfer weights
		copy(weights_[i],a.weights_[i]);
	}
}