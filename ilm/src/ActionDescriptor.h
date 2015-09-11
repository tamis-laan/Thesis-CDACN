#pragma once
#include <vector>
#include <tuple>
#include <random>

using namespace std;

/*
	The Action Descriptor class defines the actions used and is responsible for returning the constraints as wel as constructing the action vector.
*/
class ActionDescriptor
{
private:
	//Discrete actions
	vector<unsigned long int> discrete_;
	//Continues actions
	vector<pair<float,float>> real_;
	//Domain of the actions
	vector<pair<float,float>> domain_;
public:
	//Constructor
	ActionDescriptor(vector<unsigned long int> discrete, vector<pair<float,float>> real);
	//Return the dimentionality
	int action_dim() const;
	//Number of discrete actions
	int discrete_action_dim() const;
	//Number of continues actions
	int real_action_dim() const;
	//Number of actions in discrete action dimention i
	unsigned int discrete_actions_num(unsigned int i) const;
	//Return the domain
	vector<pair<float,float>> domain() const;
	//Return the domain of action i
	pair<float,float> domain(unsigned int i) const;
	//Constrain the incomming action to it's domain
	vector<float> constrain(vector<float> action) const;
	//Return defualt null action
	vector<float> null_action() const;
};