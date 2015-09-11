#include "DACN3.h"

//Compute the constrain the derivative
struct constraint_kernel
{
	bool normalized_ = false;
	float d_ = 1.0;
	constraint_kernel(float d, bool normalized) : d_(d), normalized_(normalized) {}
	template <typename Tuple> __device__ void operator()(Tuple t)
	{
		float l = -1.0; //thrust::get<3>(t);
		float u =  1.0; //thrust::get<4>(t);
		float a = (thrust::get<0>(t)<l)*(thrust::get<1>(t)<0.0)*(-1.0)*thrust::get<1>(t);
		float b = (thrust::get<0>(t)>u)*(thrust::get<1>(t)>0.0)*(-1.0)*thrust::get<1>(t);
		float c = (thrust::get<0>(t)>l)*(thrust::get<0>(t)<u)*thrust::get<1>(t);
		if(normalized_)
			thrust::get<2>(t) = d_*sign(a+b+c);
		else
			thrust::get<2>(t) = d_*(a+b+c);
	}
	inline __device__ float sign(float x)
	{
		return (x>0)-(x<0);
	}
};

//This method executes one learning step
void DACN3::step(const vector<Window> &batch)
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

	//Upload buffers to Actor
	thrust::copy(lbuffer.begin(),lbuffer.end(),  actor.input(0).data(1).begin());
	thrust::copy(rbuffer.begin(),rbuffer.end(),actor_n.input(0).data(1).begin());

	//Upload buffers to Critics
	thrust::copy(lbuffer.begin(),lbuffer.end(),  critic.input(0).data(1).begin());
	thrust::copy(rbuffer.begin(),rbuffer.end(),critic_n.input(0).data(1).begin());

	//Buffer the actions
	vector<float> action_buffer(batch_size_);
	for(int i=0; i<batch_size_; i++)
		action_buffer[i] = batch[i].actions_left()[temporal_stride_-1][0];
 
	//Forward Actor
	actor.forward(1); actor_n.forward(1);

	//Copy over Actions
	thrust::copy(  actor.output(0).data(1).begin(),  actor.output(0).data(1).end(),critic.input(1).data(1).begin());
	thrust::copy(actor_n.output(0).data(1).begin(),actor_n.output(0).data(1).end(),critic_n.input(1).data(1).begin());

	//Forward Critic
	critic.forward(1); critic_n.forward(1);

	//Save the policy based Q values
	vector<float> pQ(batch_size_);
	thrust::copy(critic.output(0).data(1).begin(),critic.output(0).data(1).end(),pQ.begin());

	//Set the Critic Delta
	cudnn_fill(1.0,critic.output(0).delta(1));

	//Backward Critic
	critic.stream(1,0).backward(1);

	//Constain/Scale and copy Derivative to Actor Delta
	thrust::for_each(	thrust::make_zip_iterator(thrust::make_tuple(critic.input(1).data(1).begin(), critic.input(1).delta(1).begin(), actor.output(0).delta(1).begin() ) ),
						thrust::make_zip_iterator(thrust::make_tuple(critic.input(1).data(1).end(),   critic.input(1).delta(1).end(),   actor.output(0).delta(1).end()   ) ),
						constraint_kernel(d_,normalized_));

	//Upload actions as Critic input
	thrust::copy(action_buffer.begin(),action_buffer.end(),critic.input(1).data(1).begin());

	//Forward Critic
	critic.stream(1,0).forward(1);

	//Compute the delta
	vector<float> critic_delta(batch_size_);
	vector<float> actor_delta(batch_size_);
	vector<float> ql(batch_size_);
	vector<float> qr(batch_size_);
	thrust::copy(  critic.output(0).data(1).begin(),  critic.output(0).data(1).end(),ql.begin());
	thrust::copy(critic_n.output(0).data(1).begin(),critic_n.output(0).data(1).end(),qr.begin());
	for(int i=0; i<batch_size_; i++)
	{
		//Get the reward 
		float r = batch[i].rewards_left()[temporal_stride_-1];
		//Get the terminal
		int   t = batch[i].terminals_left()[temporal_stride_-1];
		//Compute the Critic delta
		critic_delta[i] = (r+discount_*qr[i]*(1.0-t))*advantage_+(1.0-advantage_)*pQ[i]-ql[i];
	}

	//Copy the Critic delta to the output delta
	thrust::copy(critic_delta.begin(),critic_delta.end(),critic.output(0).delta(1).begin());

	//Backward Critic
	critic.backward(1);

	//Backward Actor
	actor.backward(1);

	//Update Critic
	critic.update(1);
	
	//Update Actor
	actor.update(1);

	//Transfer weights
	if(cycles_==0)
	{
		critic.transfer(critic_n);
		actor.transfer(actor_n);
	}
	else if(age_%cycles_==0)
	{
		critic.transfer(critic_n);
		actor.transfer(actor_n);
	}
}