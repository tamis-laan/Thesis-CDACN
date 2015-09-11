#pragma once
#include "../DACN.h"

/*
	Implementation of Gradient based Actor policy updating of the case where the Actor and Critic networks are seperated from each other. 
*/
class DACN2 final : public DACN
{
protected:
	bool normalized_;
public:
	//Constructor
	DACN2(ActionDescriptor descriptor, float discount = 0.9, int temporal_stride = 4, int capacity = 1000000, int burn_in = 1000, int batch_size = 32, float actor_learning_rate = 0.00025, float critic_learning_rate_ = 0.00025, float d = 1.0, float advantage = 0.5, unsigned int cycles = 128, bool normalized = false) : DACN(descriptor,discount,temporal_stride,capacity,burn_in,batch_size,actor_learning_rate,critic_learning_rate_,d,advantage,cycles), normalized_(normalized) {}
	//Learning step implementing gradient based actor updating
	virtual void step(const vector<Window> &batch) override;
};