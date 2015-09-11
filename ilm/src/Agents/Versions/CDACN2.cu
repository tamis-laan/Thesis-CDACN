#include "CDACN2.h"

//return value sign
inline float sign(float x)
{
	return (x>0)-(x<0);
}

//This method executes one learning step
void CDACN2::step(const vector<Window> &batch)
{
	//Allocate buffers
	vector<float> lbuffer(0);   lbuffer.reserve(width_*height_*temporal_stride_*batch_size_);
	vector<float> rbuffer(0);   rbuffer.reserve(width_*height_*temporal_stride_*batch_size_);

	//Fill buffers
	for(int i=0; i<batch_size_; i++)
	{
		auto l = batch[i].states_left();      lbuffer.insert(lbuffer.end(),l.begin(),l.end());
		auto r = batch[i].states_right();     rbuffer.insert(rbuffer.end(),r.begin(),r.end());
	}

	//Upload input states
	thrust::copy(lbuffer.begin(),lbuffer.end(),  actor_critic.input(0).data(1).begin());
	thrust::copy(rbuffer.begin(),rbuffer.end(),actor_critic_n.input(0).data(1).begin());

	//Buffer the actions
	vector<float> action_buffer(batch_size_);
	for(int i=0; i<batch_size_; i++)
		action_buffer[i] = batch[i].actions_left()[temporal_stride_-1][0];

 	//Forward Base
	  actor_critic.stream(0,0).forward(1); 
 	actor_critic_n.stream(0,0).forward(1);

	//Forward Actor
	  actor_critic.stream(1,0).forward(1); 
	actor_critic_n.stream(1,0).forward(1);

	//Copy over Actions
	thrust::copy(  actor_critic.output(0).data(1).begin(),  actor_critic.output(0).data(1).end(),  actor_critic.input(1).data(1).begin());
	thrust::copy(actor_critic_n.output(0).data(1).begin(),actor_critic_n.output(0).data(1).end(),actor_critic_n.input(1).data(1).begin());

	//Forward Critic
	  actor_critic.stream(1,1).forward(1);
	actor_critic_n.stream(1,1).forward(1);

	//Save the policy based Q values
	vector<float> pQ(batch_size_);
	thrust::copy(actor_critic.output(1).data(1).begin(),actor_critic.output(1).data(1).end(),pQ.begin());

	//Upload actions as Critic input
	thrust::copy(action_buffer.begin(),action_buffer.end(),actor_critic.input(1).data(1).begin());

	//Forward Critic
	actor_critic.stream(1,1).forward(1);

	//Compute the delta
	vector<float> critic_delta(batch_size_);
	vector<float> actor_delta(batch_size_);
	vector<float> ql(batch_size_);
	vector<float> qr(batch_size_);
	thrust::copy(  actor_critic.output(1).data(1).begin(),  actor_critic.output(1).data(1).end(),ql.begin());
	thrust::copy(actor_critic_n.output(1).data(1).begin(),actor_critic_n.output(1).data(1).end(),qr.begin());
	for(int i=0; i<batch_size_; i++)
	{
		//Get the reward 
		float r = batch[i].rewards_left()[temporal_stride_-1];
		//Get the terminal
		int   t = batch[i].terminals_left()[temporal_stride_-1];
		//Compute the Critic delta
		critic_delta[i] = (r+discount_*qr[i]*(1.0-t))*advantage_+(1.0-advantage_)*pQ[i]-ql[i];
		//Get the action
		float a = action_buffer[i];
		//Get the policy
		float mu = actor_critic.output(0).data(1)[i];
		//Compute the Actor delta
		if(normalized_)
			actor_delta[i] = d_*(critic_delta[i]>0)*sign(a-mu);
		else
			actor_delta[i] = d_*(critic_delta[i]>0)*critic_delta[i]*sign(a-mu);
	}

	//Copy the Actor delta
	thrust::copy(actor_delta.begin(),actor_delta.end(),actor_critic.output(0).delta(1).begin());
	
	//Copy the Critic delta to the output delta
	thrust::copy(critic_delta.begin(),critic_delta.end(),actor_critic.output(1).delta(1).begin());

	//Backward
	actor_critic.backward(1);
	
	//Update
	actor_critic.update(1);

	//Transfer weights
	if(cycles_==0)
		actor_critic.transfer(actor_critic_n);
	else if(age_%cycles_==0)
		actor_critic.transfer(actor_critic_n);
}