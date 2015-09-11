#pragma once

#include "Memory.h"
#include "ActionDescriptor.h"

#include <vector>
#include <tuple>
using namespace std;


/*
	The agent class is a virtual interface used to implement any sort of Replay Memory Agent. The virtual methods policy and step should be overidden in order to 
	implement the new agent. The policy executes the agent policy and the step method executes one training step.
*/
class Agent
{ 
protected:
	int temporal_stride_ = 0;
	int width_           = 0;
	int height_          = 0;
	int capacity_        = 0;
	int burn_in_         = 0;
	int batch_size_      = 0;
	unsigned long age_   = 0;

	//Action Descriptor
	ActionDescriptor descriptor_;
	//Current state window
	Window window_;
	//Replay Memory
	Memory memory_;
	//Agents policy (virtual)
	virtual pair<vector<float>,float> policy(const Window &window) = 0;
	//Agents learning mechanism
	virtual void step(const vector<Window> &batch) = 0;
	//Random number generator
	mt19937 engine;
public:
	//Constructor
	Agent(ActionDescriptor descriptor, int temporal_stride, int width, int height, int capacity, int burn_in, int batch_size);
	//Return the age of the agent
	unsigned long age() const;
	//Gather state data and execute policy
	pair<vector<float>,float> forward(const vector<unsigned char> &state);
	//Gather reward and terminal data
	void backward(float reward, bool terminal);	
	//do a learning step
	void update();
	//Reset
	void reset();
	//Evaluate current policy 
	pair<vector<float>,float> evaluate(const vector<unsigned char> &state);
};