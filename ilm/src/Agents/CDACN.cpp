#include "CDACN.h"
#include <fstream>
#include <limits>



//Define the neural network used
ActorCritic::ActorCritic(cudnnHandle_t handle, std::vector<int> batch, int temporal_stride, int action_dim, float base_learning_rate, float actor_learning_rate, float critic_learning_rate) : Network(handle,batch)
{
	//Input state
	auto s  = create<Block>(temporal_stride,84,84);
	
	//Input action for Critic
	auto a  = create<Block>(action_dim);

	//Output Q value produce by Critic
	auto q  = create<Block>(1);

	//Output policy produced by Actor
	auto p  = create<Block>(action_dim);

	//Base
	auto Bl1    = create<Convolution>(64,8,8,4,4,0.01,0.01,4,4);
	auto Bb2    = create<Block>(64,22,22);
	auto Bl2    = create<Activation>(Activation::Mode::LReLU);
	auto Bb3    = create<Block>(64,22,22);
	auto Bl3    = create<Convolution>(64,4,4,2,2,0.01,0.01);
	auto Bb4    = create<Block>(64,10,10);
	auto Bl4    = create<Activation>(Activation::Mode::LReLU);
	auto Bb5    = create<Block>(64,10,10);
	auto Bl5    = create<Convolution>(64,3,3,1,1,0.01,0.01);
	auto Bb6    = create<Block>(64,8,8);
	auto Bl6    = create<Activation>(Activation::Mode::LReLU);
	auto Bb7    = create<Block>(64,8,8);
	auto Bl7    = create<Dense>(256,0.01,0.01);
	auto Bb8    = create<Block>(256);
	auto Bl8    = create<Activation>(Activation::Mode::LReLU);
	auto Bb9    = create<Block>(256);
	auto Bl9    = create<Split>();
	auto Bb10_1 = create<Block>(256);
	auto Bb10_2 = create<Block>(256);
	auto base   = create<Stream>(ADAM(base_learning_rate),s,Bl1,Bb2,Bl2,Bb3,Bl3,Bb4,Bl4,Bb5,Bl5,Bb6,Bl6,Bb7,Bl7,Bb8,Bl8,Bb9,Bl9,vector<Block>{Bb10_1,Bb10_2});
	
	//Actor
	auto Al9   = create<Dense>(256,0.01,0.01);
	auto Ab10  = create<Block>(256);
	auto Al10  = create<Activation>(Activation::Mode::LReLU);
	auto Ab11  = create<Block>(256);
	auto Al11  = create<Dense>(128,0.01,0.01);
	auto Ab12  = create<Block>(128);
	auto Al12  = create<Activation>(Activation::Mode::LReLU);
	auto Ab13  = create<Block>(128);
	auto Al13  = create<Dense>(64);
	auto Ab14  = create<Block>(64);
	auto Al14  = create<Activation>(Activation::Mode::LReLU);
	auto Ab15  = create<Block>(64);
	auto Al15  = create<Dense>(action_dim,0.01,0.0);
	auto actor = create<Stream>(ADAM(actor_learning_rate),Bb10_1,Al9,Ab10,Al10,Ab11,Al11,Ab12,Al12,Ab13,Al13,Ab14,Al14,Ab15,Al15,p);

	//Critic
	auto ml       = create<Merge>();
	auto Cb9_     = create<Block>(256+action_dim);
	auto Cl9      = create<Dense>(256,0.01,0.01);
	auto Cb10     = create<Block>(256);	
	auto Cl10     = create<Activation>(Activation::Mode::LReLU);
	auto Cb11     = create<Block>(256);
	auto Cl11     = create<Dense>(128,0.01,0.01);
	auto Cb12     = create<Block>(128);
	auto Cl12     = create<Activation>(Activation::Mode::LReLU);
	auto Cb13     = create<Block>(128);
	auto Cl13     = create<Dense>(64);
	auto Cb14     = create<Block>(64);
	auto Cl14     = create<Activation>(Activation::Mode::LReLU);
	auto Cb15     = create<Block>(64);
	auto Cl15     = create<Dense>(1,0.01,0.0);
	auto critic = create<Stream>(ADAM(critic_learning_rate),vector<Block>{Bb10_2,a},ml,Cb9_,Cl9,Cb10,Cl10,Cb11,Cl11,Cb12,Cl12,Cb13,Cl13,Cb14,Cl14,Cb15,Cl15,q);

	//Push the streams
	push(base);
	push(actor,critic);
	
	//Set inputs
	inputs(s,a);
	
	//Set outputs
	outputs(p,q);
}

//Constructor
CDACN::CDACN(ActionDescriptor descriptor, float discount, int temporal_stride, int capacity, int burn_in, int batch_size, float base_learning_rate, float actor_learning_rate, float critic_learning_rate, float d, float advantage, unsigned int cycles) : 
			Agent(descriptor,temporal_stride,84,84,capacity,burn_in,batch_size),
			discount_(discount),
			exploration_rate_(1.0),
			uniform_real(0.0,1.0),
			base_learning_rate_(base_learning_rate),
			actor_learning_rate_(actor_learning_rate),
			critic_learning_rate_(critic_learning_rate),
			d_(d),
			advantage_(advantage),
			cycles_(cycles)
{
	cudnnCreate(&handle);
	actor_critic   = ActorCritic(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),base_learning_rate_,actor_learning_rate_,critic_learning_rate_);
	actor_critic_n = ActorCritic(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),base_learning_rate_,actor_learning_rate_,critic_learning_rate_);
	//Sync parameters
	actor_critic.transfer(actor_critic_n);
}

//Scan the network and log it into scan.csv
void CDACN::scan(float lower, float upper, unsigned int n)
{
	std::cout << "[!] Making scan!" << std::endl;

	ofstream file("scan.csv");
	for(int i=0; i<n+1; i++)
	{
		//Construct action
		float a = lower+(i*(upper-lower)/n);
		
		//Set action
		actor_critic.input(1).data(0)[0] = a;
		
		//Forward critic
		actor_critic.stream(1,1).forward(0);

		//Get the Q value
		float q = actor_critic.output(1).data(0)[0];

		//Set delta value
		actor_critic.output(1).delta(0)[0] = 1.0;
		
		//Backward Q value
		actor_critic.stream(1,1).backward(0);
		
		//Get the action derivative
		float da = actor_critic.input(1).delta(0)[0];
		
		//Write to file
		file << a << " , " << q << " , " << da << endl;
	}
	
	//Forward the Actor
	// actor.forward(0); //Redundant really
	
	//Get policy
	float mu = actor_critic.output(0).data(0)[0];

	//Write to file
	file << ",,," << mu << endl;
}

//This method executes the policy
pair<vector<float>,float> CDACN::policy(const Window &window)
{
	if(uniform_real(engine)<exploration_rate_ or not window.full())
		return explore();
	else
		return exploit(window);
}

//Executes explore step
pair<vector<float>,float> CDACN::explore()
{
	vector<float> action;
	for(int i=0; i<descriptor_.discrete_action_dim(); i++)
	{
		uniform_int_distribution<> dist(0,descriptor_.discrete_actions_num(i)-1);
		action.push_back( static_cast<float>( dist(engine) ) );
	}
	for(int i=0; i<descriptor_.real_action_dim(); i++)
	{
		uniform_real_distribution<> dist(descriptor_.domain(i).first,descriptor_.domain(i).second);
		action.push_back( static_cast<float>( dist(engine) ) );
	}
	return make_pair(action,numeric_limits<float>::quiet_NaN()); 
}

//Executes exploitation step
pair<vector<float>,float> CDACN::exploit(const Window &window)
{
	//Extract the current state
	vector<float> state = window.states_right();
	//Upload state to Actor Critic
	thrust::copy(state.begin(),state.end(),actor_critic.input(0).data(0).begin());
	//Forward the Base
	actor_critic.stream(0,0).forward(0);
	//Forward the Actor
	actor_critic.stream(1,0).forward(0);
	//Create the action vector
	vector<float> action(descriptor_.action_dim());
	//Get the action from the Actor Critic
	thrust::copy(actor_critic.output(0).data(0).begin(), actor_critic.output(0).data(0).end(), action.begin());
	//Set the action
	thrust::copy(actor_critic.output(0).data(0).begin(), actor_critic.output(0).data(0).end(), actor_critic.input(1).data(0).begin() );
	//Forward the Critic
	actor_critic.stream(1,1).forward(0);
	//Get the Q value
	float q = actor_critic.output(1).data(0)[0];
	//Return the action
	return make_pair(action,q);
}