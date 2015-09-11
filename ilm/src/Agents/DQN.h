#pragma once

#include "../Agent.h"
#include "Network.h"
#include "Solvers/ADAM.h"

//This class defines the neural network topology
class DQN_N final : public ddnet::Network
{
public:
	DQN_N() = default;
	DQN_N(cudnnHandle_t handle, vector<int> batch, unique_ptr<ddnet::Solver> solver, int temporal_stride, int number_of_actions);
};

/*
	Implementation of the Deep Q Network Agent described in the Deepmind paper, Human Level Control...
*/
class DQN final : public Agent
{
private:
	//Device handle for network
	cudnnHandle_t handle = NULL;
	//Deep Q Network
	DQN_N network;
	//Discount factor
	float discount_;
	//Exploration Rate
	float exploration_rate_;
	//Random Number generator
	mt19937 engine;
	//Uniform distribution
	uniform_int_distribution<> uniform_discrete;
	//Uniform distribution
	uniform_real_distribution<> uniform_real;
	//Counter
	unsigned long counter = 0;
	//Frequency with which we backprop (based on counter)
	const unsigned long update_frequency = 1;
	//Exploration routine
	vector<float> explore();
	//Exploitation routine
	vector<float> exploit(const Window &window);

public:
	//Constructor
	DQN(ActionDescriptor descriptor, float discount = 0.9, int temporal_stride = 4, int capacity = 1000000, int burn_in = 1000, int batch_size = 32);
	//Set exploration rate
	void exploration_rate(float r) {exploration_rate_=r;}
	virtual vector<float> policy(const Window &window) override;
	virtual void step(const vector<Window> &batch) override;
};