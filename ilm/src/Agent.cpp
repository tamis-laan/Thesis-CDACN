#include "Agent.h"
#include <limits>

//Constructor
Agent::Agent(ActionDescriptor descriptor, int temporal_stride, int width, int height, int capacity, int burn_in, int batch_size) : 
	descriptor_(descriptor),
	temporal_stride_(temporal_stride), 
	width_(width),
	height_(height),
	capacity_(capacity), 
	burn_in_(burn_in),
	batch_size_(batch_size),
	window_(temporal_stride,width,height,descriptor.action_dim()),
	memory_(capacity),
	engine((unsigned)time(0))
{}

//Return the age of the agent
unsigned long Agent::age() const 
{
	return age_;
}

//Agents policy (virtual)
pair<vector<float>,float> Agent::forward(const std::vector<unsigned char> &state)
{
	//Push the state
	window_.push_state(state);

	//Compute action if window is full
	pair<vector<float>,float> package;
	if(window_.full())
		package = policy(window_);
	else
		package = make_pair(descriptor_.null_action(),numeric_limits<float>::quiet_NaN());

	//Push the action in the window
	window_.push_action(package.first);

	//Constrain actions
	package.first = descriptor_.constrain(package.first);

	//Return the actions
	return move(package);
}

//Gather reward and terminal data and do a learning step
void Agent::backward(float reward, bool terminal)
{
	//Push reward and terminal
	window_.push_reward(reward);
	window_.push_terminal(terminal);

	//True terminal requires stub
	if(terminal)
		window_.push_stub();

	//Add window to database
	memory_.add(window_);

	//Reset the window
	if(terminal)
		window_.reset();

	//Update the age counter
	age_++;
}

//Do a learning step
void Agent::update()
{
	if(memory_.size()>batch_size_ and age_>burn_in_)
	{
		uniform_int_distribution<> uniform(0,memory_.size()-batch_size_);
		step( memory_.get(uniform(engine),batch_size_) );
	}
}

//Simply reset the window
void Agent::reset()
{
	window_.reset();
}

//
pair<vector<float>,float> Agent::evaluate(const std::vector<unsigned char> &state)
{
	//Push the state
	window_.push_state(state);

	//Compute action if window is full
	pair<vector<float>,float> package;
	if(window_.full())
		package = policy(window_);
	else
		package = make_pair(descriptor_.null_action(),numeric_limits<float>::quiet_NaN());

	//Constrain the action
	package.first = descriptor_.constrain(package.first);

	//Push the action in the window
	window_.push_action(package.first);

	//Push reward and terminal
	window_.push_reward(0.0);
	window_.push_terminal(false);

	//Return the package
	return move(package);
}
