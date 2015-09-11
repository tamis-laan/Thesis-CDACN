#include "Network.h"
#include "Util.h"

namespace ddnet
{
	//Constructor
	Stream::Stream(std::shared_ptr<Solver> solver, std::vector<Adapter> adapters) : solver_(solver), adapters_(adapters) {}

	//Get the solver
	std::shared_ptr<Solver> Stream::getSolver()
	{
		return solver_;
	}

	//Forward input batch trough stream
	void Stream::forward(int batch)
	{
		for(auto itr = adapters_.begin(); itr != adapters_.end(); ++itr)
			itr->element().forward(batch);
	}

	//Backward stream
	void Stream::backward(int batch)
	{
		for(auto itr = adapters_.rbegin();  itr != adapters_.rend(); ++itr) 
			itr->element().backward(batch);
	}

	//Update Stream
	void Stream::update(int batch)
	{
		for(auto itr = adapters_.begin(); itr != adapters_.end(); ++itr)
		{
			itr->element().update(batch,false);
			solver_->step(*itr);
		}
	}

	//Transfer weights between streams
	void Stream::transfer(Stream &s)
	{
		//Check equal number of adapters
		if(adapters_.size() != s.adapters_.size()) throw ddnet_exception(10);
		//Transfer weights between adapters
		for(int i=0; i<adapters_.size(); i++)
			adapters_[i].transfer(s.adapters_[i]);
	}

	//Constructor
	Network::Network(cudnnHandle_t handle, std::vector<int> batch) : handle_(handle), batch_(batch) {}

	//Get stream refrence
	Stream& Network::stream(int i, int j)
	{
		return streams_[i][j];
	}

	//Get an input block
	Block& Network::input(int i)
	{
		return inputs_[i];
	}

	//Get an output block
	Block& Network::output(int i)
	{
		return outputs_[i];
	}

	//Forward streams
	void Network::forward(int batch)
	{
		for(auto itr = streams_.begin(); itr != streams_.end(); ++itr)
			for(auto &stream : *itr)
				stream.forward(batch);
	}

	//Backward streams
	void Network::backward(int batch)
	{
		for(auto itr = streams_.rbegin(); itr != streams_.rend(); ++itr)
			for(auto &stream : *itr)
				stream.backward(batch);
	}

	//Update streams
	void Network::update(int batch)
	{
		for(auto itr = streams_.begin(); itr != streams_.end(); ++itr)
			for(auto &stream : *itr)
				stream.update(batch);
	}

	//Transfer weights to identical network
	void Network::transfer(Network &n)
	{
		//Check equal stream depth
		if(streams_.size()!=n.streams_.size()) throw ddnet_exception(10);
		//Loop
		for(int i=0; i<streams_.size(); i++)
		{
			//Check equal stream width
			if(streams_[i].size() != n.streams_[i].size()) throw ddnet_exception(10);
			//Transfer weights
			for(int j=0; j<streams_[i].size(); j++)
				streams_[i][j].transfer(n.streams_[i][j]);
		}
	}
}