#pragma once
#include "Network.h"
#include "../Agent.h"
using namespace ddnet;
using namespace std;

class Actor : public Network
{
public:
	Actor() = default;
	Actor(cudnnHandle_t handle, std::vector<int> batch, int temporal_stride, int action_dim, float learning_rate);
};

class Critic : public Network
{
public:
	Critic() = default;
	Critic(cudnnHandle_t handle, std::vector<int> batch, int temporal_stride, int action_dim, float learning_rate);
};

class DACN : public Agent
{
protected:
	//Device handle for network
	cudnnHandle_t handle = NULL;
	//Deep Actor Critic Network
	Actor  actor;
	Actor  actor_n;
	Critic critic;
	Critic critic_n;
	//Discount factor
	float discount_;
	//Exploration Rate
	float exploration_rate_;
	//Critic Learning Rate
	float critic_learning_rate_;
	//Actor Learning Rate
	float actor_learning_rate_;
	//Actor Difference Rate
	float d_;
	//Advantage 
	float advantage_;
	//Cycles
	unsigned int cycles_;
	//Random Number generator
	mt19937 engine;
	//Uniform distribution
	uniform_real_distribution<> uniform_real;
	//Exploration routine
	pair<vector<float>,float> explore();
	//Exploitation routine
	pair<vector<float>,float> exploit(const Window &window);
public:
	//Constructor
	DACN(ActionDescriptor descriptor, float discount = 0.94, int temporal_stride = 4, int capacity = 1000000, int burn_in = 1000, int batch_size = 32, float actor_learning_rate = 0.000025, float critic_learning_rate_ = 0.000025, float d = 1.0, float advantage_ = 1.0, unsigned int cycles = 256);
	//Set exploration rate
	void exploration_rate(float r) {exploration_rate_=r;}
	//Get exploration rate
	float exploration_rate() {return exploration_rate_;}
	//Policy method
	virtual pair<vector<float>,float> policy(const Window &window) override;
	//Scan the network
	void scan(float lower = -1.2, float upper = 1.2, unsigned int n = 180);
};