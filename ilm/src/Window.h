#pragma once
#include <vector>
#include <boost/circular_buffer.hpp>
#include "lz4.h"
using namespace std;

/*
	The window class encapsulates information required by the agent class to learn it´s network. Note that states, actions, rewards and terminals should be pushed in the 
	following order:
		1. push_state
		2. push_action
		3. push_reward
		4. push_terminal
	If this is not the case the window can go out of sync causing weird behavior
*/
class Window final
{
protected:
	//Holds temporal_stride number of states (84x84) for atari
	boost::circular_buffer<vector<unsigned char>> states_;
	//Holds temporal_stride number of multi-dimentional actions 
	boost::circular_buffer<vector<float>>         actions_;
	//Holds temporal_stride number of rewards
	boost::circular_buffer<float>                 rewards_;
	//Holds temporal_stride number of terminals (indicates end of episode)
	boost::circular_buffer<int>                   terminals_;

	//Time dimention size
	int temporal_stride_;
	//State width
	int state_width_;
	//State height
	int state_height_;
	//Dimentionality of action
	int action_dim_;
	
	//Stub is used for padding in null state 
	vector<unsigned char> state_stub_;

public:
	//Constructor
	Window(int temporal_stride, int state_width, int state_height, int action_dim);
	//This method pushes a null state (using stub) in order to mark end of episode
	void push_stub();
	//Push a new state into the window
	void push_state(const vector<unsigned char> &state);
	//Push a new action
	void push_action(const vector<float> &action);
	//Push a new reward
	void push_reward(float reward);
	//Push a new terminal
	void push_terminal(int terminal);
	//Resets the states, actions, rewards and terminals 
	void reset();
	//Return the number of pushed elements in the window
	int size() const;
	//Return the capacity of the window
	int capacity() const;
	//Check if the window is full
	bool full() const;
	//Return the left state
	vector<float> states_left() const;
	//Return the right states
	vector<float> states_right() const;
	//Return the left actions
	vector<vector<float>> actions_left() const;
	//Return the right actions
	vector<vector<float>> actions_right() const;
	//Return left rewards
	vector<float> rewards_left() const;
	//Return right rewards
	vector<float> rewards_right() const;
	//Return left terminals
	vector<int> terminals_left() const;
	//Return right terminals
	vector<int> terminals_right() const;
};
