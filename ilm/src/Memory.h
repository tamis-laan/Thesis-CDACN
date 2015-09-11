#pragma once
#include "Window.h"

#include <vector>
using namespace std;

/*
	The database keeps the experiences together in the form of a vector of Windows.
*/
class Memory final
{
protected:
	//Memory of windows
	vector<Window> data_;
	//Random number generator
	mt19937 engine;
	//Uniform distribution
	uniform_int_distribution<> uniform;
public:
	//Constructor
	Memory(int capacity);
	//Copy not allowed 
	Memory& operator=(const Memory &s) = delete;
	//Add a new window to the database
	void add(const Window &window);
	//Get sample from the data_ (WARNING: You can easily sample out of valid range) (NOTE: In the original paper they sample randomly!)
	vector<Window> const get(int index, int batch_size) const;
	//Return Memory capacity	
	int capacity() const;
	//Return Memory size	
	int size() const;
};
