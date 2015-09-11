#pragma once
#include "Util.h"

//Ball container
class Ball
{
public:
	VEC2 position = {0.0, 0.0     };
	float size    = 0.05;
	VEC2 velocity = {0.0, 0.0     };
	COLR color    = {1.0, 1.0, 1.0};
	void draw();
	float logic();
};

//Paddle Container
class Paddle
{
public:
	float position_0 = 0;
	float position_1 = 0;
	VEC2 size        = {0.3, 0.05     };
	COLR color       = {0.75, 0.75 , 0.75};
	void draw();
	void move(float a);
	float logic(Ball &ball);
};

//Generic wall container
class Wall
{
private:
	VEC2 position;
	VEC2 size;
	COLR color;
	float reward;
public:
	Wall(float x, float y, float w, float h, float reward = 0.0, float r = 0.5, float g = 0.5, float b = 0.5);
	void draw();
	float logic(Ball &ball);
};
