#pragma once
#include <vector>
#include <tuple>
#include <random>
#include "Environment.h"
#include "Objects.h"

using namespace std;

class Bounce : public Environment
{
protected:
	//Basic elements
	Paddle paddle;
	Ball ball;
	Wall left;
	Wall right;
	Wall top;

	//Variable elements
	vector<Wall> obstacles; 

	//Book keeping 
	float score   = 0;

	//Random Number generator
	std::mt19937 engine;
public:

	//Constructor
	Bounce();

	//Reset
	void reset();

	//Compute game logic
	float logic(unsigned int steps = 4);

	//Draw the game
	void draw();

	virtual float step(std::vector<float> action);
};