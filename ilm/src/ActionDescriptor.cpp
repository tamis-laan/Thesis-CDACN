#include "ActionDescriptor.h"

#include <iostream>

using namespace std;

//Constructor
ActionDescriptor::ActionDescriptor(vector<unsigned long int> discrete, vector<pair<float,float>> real) : discrete_(discrete), real_(real), domain_(discrete.size()+real.size())
{
	//Fill the domain vector using the discrete vec
	for(int i=0; i<discrete_.size(); i++)
	{
		if(discrete_[i]==0) throw "MUST HAVE ATLEAST ONE ACTION!";
		domain_[i].first  = 0.0;
		domain_[i].second = static_cast<float>(discrete_[i]-1);
	}
	//Fill the domain vector using the real vec
	for(int i=0; i<real_.size(); i++)
	{
		domain_[i+discrete_.size()].first  = real_[i].first;
		domain_[i+discrete_.size()].second = real_[i].second;
	}
}

//Return the dimentionality
int ActionDescriptor::action_dim() const
{
	return discrete_.size() + real_.size();
}

//Number of discrete actions
int ActionDescriptor::discrete_action_dim() const
{
	return discrete_.size();
}

//Number of continues actions
int ActionDescriptor::real_action_dim() const
{
	return real_.size();
}

//Number of actions in discrete action dimention i
unsigned int ActionDescriptor::discrete_actions_num(unsigned int i) const
{
	return discrete_[i];
}

vector<pair<float,float>> ActionDescriptor::domain() const
{
	return domain_;
}

//Return the domain of action i
pair<float,float> ActionDescriptor::domain(unsigned int i) const
{
	return domain_[i];
}

//Tile the incomming action
vector<float> ActionDescriptor::constrain(vector<float> action) const
{

	//Loop over the discrete actions and tile
	for(int i=0; i<discrete_.size(); i++)
	{
		float a = (action[i]<domain_[i].first)*domain_[i].first;
		float b = (action[i]>domain_[i].second)*domain_[i].second;
		float c = (action[i]>=domain_[i].first and action[i]<=domain_[i].second)*static_cast<float>(std::round(action[i]));
		action[i] = a + b + c;
	}

	//Loop over the continues actions and tile
	for(int i=discrete_.size(); i<discrete_.size()+real_.size(); i++)
	{
		float a = (action[i]<domain_[i].first)*domain_[i].first;
		float b = (action[i]>domain_[i].second)*domain_[i].second;
		float c = (action[i]>domain_[i].first)*(action[i]<domain_[i].second)*action[i];
		action[i] = a + b + c;
	}

	//Return the action
	return action;
}

vector<float> ActionDescriptor::null_action() const
{
	int dim = action_dim();
	vector<float> action(dim);
	for(int i=0; i<dim ; i++)
		action[i]=domain_[i].first + (domain_[i].second-domain_[i].first);
	return action;
}