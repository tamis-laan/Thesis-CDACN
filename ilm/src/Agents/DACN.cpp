#include "DACN.h"
#include <fstream>
#include <limits>

Actor::Actor(cudnnHandle_t handle, std::vector<int> batch, int temporal_stride, int action_dim, float learning_rate) : Network(handle,batch)
{
	//Input state
	auto s = create<Block>(temporal_stride,84,84);

	//Output policy produced by Actor
	auto p = create<Block>(action_dim);

	//Actor
	auto Al1   = create<Convolution>(64,8,8,4,4,0.01,0.01,4,4);
	auto Ab2   = create<Block>(64,22,22);
	auto Al2   = create<Activation>(Activation::Mode::LReLU);
	auto Ab3   = create<Block>(64,22,22);
	auto Al3   = create<Convolution>(64,4,4,2,2,0.01,0.01);
	auto Ab4   = create<Block>(64,10,10);
	auto Al4   = create<Activation>(Activation::Mode::LReLU);
	auto Ab5   = create<Block>(64,10,10);
	auto Al5   = create<Convolution>(64,3,3,1,1,0.01,0.01);
	auto Ab6   = create<Block>(64,8,8);
	auto Al6   = create<Activation>(Activation::Mode::LReLU);
	auto Ab7   = create<Block>(64,8,8);
	auto Al7   = create<Dense>(256,0.01,0.01);
	auto Ab8   = create<Block>(256);
	auto Al8   = create<Activation>(Activation::Mode::LReLU);
	auto Ab9   = create<Block>(256);	
	auto Al9   = create<Dense>(256,0.01,0.01);
	auto Ab10  = create<Block>(256);
	auto Al10  = create<Activation>(Activation::Mode::LReLU);
	auto Ab11  = create<Block>(256);
	auto Al11  = create<Dense>(128,0.01,0.01);
	auto Ab12  = create<Block>(128);
	auto Al12  = create<Activation>(Activation::Mode::LReLU);
	auto Ab13  = create<Block>(128);

	auto Al13  = create<Dense>(64,0.01,0.01);
	auto Ab14  = create<Block>(64);
	auto Al14  = create<Activation>(Activation::Mode::LReLU);
	auto Ab15  = create<Block>(64);

	auto Al15  = create<Dense>(action_dim,0.01,0.0);
	auto actor = create<Stream>(ADAM(learning_rate,0.95,0.95),s,Al1,Ab2,Al2,Ab3,Al3,Ab4,Al4,Ab5,Al5,Ab6,Al6,Ab7,Al7,Ab8,Al8,Ab9,Al9,Ab10,Al10,Ab11,Al11,Ab12,Al12,Ab13,Al13,Ab14,Al14,Ab15,Al15,p);
	 
	//Push the streams
	push(actor);
	
	//Set inputs
	inputs(s);
	
	//Set outputs
	outputs(p);
}

Critic::Critic(cudnnHandle_t handle, std::vector<int> batch, int temporal_stride, int action_dim, float learning_rate) : Network(handle,batch)
{
	//Input state
	auto s = create<Block>(temporal_stride,84,84);
	
	//Input action for Critic
	auto a = create<Block>(action_dim);

	//Output Q value produce by Critic
	auto q = create<Block>(1);

	//Critic
	auto Cl1      = create<Convolution>(64,8,8,4,4,0.01,0.01,4,4);
	auto Cb2      = create<Block>(64,22,22);
	auto Cl2      = create<Activation>(Activation::Mode::LReLU);
	auto Cb3      = create<Block>(64,22,22);
	auto Cl3      = create<Convolution>(64,4,4,2,2,0.01,0.01);
	auto Cb4      = create<Block>(64,10,10);
	auto Cl4      = create<Activation>(Activation::Mode::LReLU);
	auto Cb5      = create<Block>(64,10,10);
	auto Cl5      = create<Convolution>(64,3,3,1,1,0.01,0.01);
	auto Cb6      = create<Block>(64,8,8);
	auto Cl6      = create<Activation>(Activation::Mode::LReLU);
	auto Cb7      = create<Block>(64,8,8);
	auto Cl7      = create<Dense>(256,0.01,0.01);
	auto Cb8      = create<Block>(256);
	auto Cl8      = create<Activation>(Activation::Mode::LReLU);
	auto Cb9      = create<Block>(256); 
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

	auto Cl13     = create<Dense>(64,0.01,0.01);
	auto Cb14     = create<Block>(64);
	auto Cl14     = create<Activation>(Activation::Mode::LReLU);
	auto Cb15     = create<Block>(64);

	auto Cl15     = create<Dense>(1,0.01,0.0); 
	auto critic_1 = create<Stream>(ADAM(learning_rate,0.95,0.95),s,Cl1,Cb2,Cl2,Cb3,Cl3,Cb4,Cl4,Cb5,Cl5,Cb6,Cl6,Cb7,Cl7,Cb8,Cl8,Cb9);
	auto critic_2 = create<Stream>(ADAM(learning_rate,0.95,0.95),vector<Block>{Cb9,a},ml,Cb9_,Cl9,Cb10,Cl10,Cb11,Cl11,Cb12,Cl12,Cb13,Cl13,Cb14,Cl14,Cb15,Cl15,q);
	
	//Push the streams
	push(critic_1);
	push(critic_2);

	//Set inputs
	inputs(s,a);

	//Set outputs
	outputs(q);
}

//Constructor
DACN::DACN(ActionDescriptor descriptor, float discount, int temporal_stride, int capacity, int burn_in, int batch_size, float actor_learning_rate, float critic_learning_rate, float d, float advantage, unsigned int cycles) : 
			Agent(descriptor,temporal_stride,84,84,capacity,burn_in,batch_size),
			discount_(discount),
			exploration_rate_(1.0),
			uniform_real(0.0,1.0),
			actor_learning_rate_(actor_learning_rate),
			critic_learning_rate_(critic_learning_rate),
			d_(d),
			advantage_(advantage),
			cycles_(cycles)
{
	cudnnCreate(&handle);
	actor    =  Actor(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),actor_learning_rate_);
	actor_n  =  Actor(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),actor_learning_rate_);
	critic   = Critic(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),critic_learning_rate_);
	critic_n = Critic(handle,{1,batch_size},temporal_stride,descriptor_.action_dim(),critic_learning_rate_);
	//Sync parameters
	critic.transfer(critic_n);
	actor.transfer(actor_n);
}

//Scan the network and log it into scan.csv
void DACN::scan(float lower, float upper, unsigned int n)
{
	std::cout << "[!] Making scan!" << std::endl;

	ofstream file("scan.csv");
	for(int i=0; i<n+1; i++)
	{
		//Construct action
		float a = lower+(i*(upper-lower)/n);
		
		//Set action
		critic.input(1).data(0)[0] = a;
		
		//Forward critic
		critic.forward(0);

		//Get the Q value
		float q  = critic.output(0).data(0)[0];

		//Set delta value
		critic.output(0).delta(0)[0] = 1.0;
		
		//Backward Q value
		critic.backward(0);
		
		//Get the action derivative
		float da = critic.input(1).delta(0)[0];
		
		//Write to file
		file << a << " , " << q << " , " << da << endl;
	}
	
	//Forward the Actor
	actor.forward(0); //Redundant really
	
	//Get policy
	float mu = actor.output(0).data(0)[0];

	//Write to file
	file << ",,," << mu << endl;
}

//This method executes the policy
pair<vector<float>,float> DACN::policy(const Window &window)
{
	if(uniform_real(engine)<exploration_rate_ or not window.full())
		return explore();
	else
		return exploit(window);
}

//Executes explore step
pair<vector<float>,float> DACN::explore()
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
pair<vector<float>,float> DACN::exploit(const Window &window)
{
	//Extract the current state
	vector<float> state = window.states_right();
	//Upload state to Actor
	thrust::copy(state.begin(),state.end(),actor.input(0).data(0).begin());
	//Upload state to Critic
	thrust::copy(state.begin(),state.end(),critic.input(0).data(0).begin());
	//Forward the Actor
	actor.forward(0);
	//Create the action vector
	vector<float> action(descriptor_.action_dim());
	//Get the action from the Actor
	thrust::copy(actor.output(0).data(0).begin(), actor.output(0).data(0).end(), action.begin());
	//Set the action
	thrust::copy(actor.output(0).data(0).begin(), actor.output(0).data(0).end(), critic.input(1).data(0).begin() );
	//Forward the Critic
	critic.forward(0);
	//Get the Q value
	float q = critic.output(0).data(0)[0];
	//Return the action
	return make_pair(action,q);
}