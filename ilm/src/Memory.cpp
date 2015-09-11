#include "Memory.h"


//Constructor
Memory::Memory(int capacity) : engine((unsigned)time(0)), uniform(0,capacity-1) 
{
	data_.reserve(capacity);
}

//Add a new window to the database in random order.
void Memory::add(const Window &window)
{
	if(window.full())
	{
		if( data_.size() == data_.capacity() )
			data_[uniform(engine)] = window;
		else
		{
			data_.push_back(window);
			std::uniform_int_distribution<> uni(0,data_.size()-1);
			std::swap(data_[uni(engine)],data_[data_.size()-1]);
		}
	}
}

//Get sample from the data_ (WARNING: You can easily sample out of valid range) (NOTE: In the original paper they sample randomly!)
vector<Window> const Memory::get(int index, int batch_size) const
{
	return move( vector<Window>(&data_[index],&data_[index]+batch_size) );
}

//Return Memory capacity	
int Memory::capacity() const 
{
	return data_.capacity();
}

//Return Memory size	
int Memory::size() const 
{
	return data_.size();
}