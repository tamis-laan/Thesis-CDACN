#define CATCH_CONFIG_MAIN 
#include <catch.h>
#include "./src/Window.h"
#include "./src/Memory.h"

TEST_CASE( "Class Window tests", "[Window]" ) 
{
	Window window = Window(3,4,4,3);
	vector<unsigned char> state = { 'a','a','b','c',  'd','d','d','d',  'd','d','d','d',  'e','e','e','f'};

	REQUIRE(window.full() == false);
	window.push_stub();
	window.push_stub();
	window.push_stub();
	window.push_state(state);
	REQUIRE(window.full() == true);
	
	auto states = window.states_right();
	REQUIRE(states[4*4*3-1] == 102);
	REQUIRE(states[4*4*3-2] == 101);
	REQUIRE(states[4*4*3-5] == 100);
	states = window.states_left();
	REQUIRE(states[4*4*3-1] == 120);

	vector<float> action = {1.0f,2.0f,3.0f};
	window.push_action(action);
	auto actions = window.actions_right();
	REQUIRE(actions[3-1][0] == 1.0f   );
	REQUIRE(actions[3-1][2] == 3.0f   );
	REQUIRE(actions[2-1][0] == -1337.0f);

	actions = window.actions_left();
	REQUIRE(actions[3-1][0] == -1337.0f);

	window.push_reward(9999.0f);
	vector<float> rewards = window.rewards_right();
	REQUIRE(rewards[3-1] == 9999.0f);
	REQUIRE(rewards[3-2] == -1337.0f);

	window.push_terminal(true);
	vector<int> terminals = window.terminals_right();
	REQUIRE(terminals[3-1] == true);
	REQUIRE(terminals[3-2] == -1337);

	REQUIRE(window.size()     == 4);
	REQUIRE(window.capacity() == 4);
}