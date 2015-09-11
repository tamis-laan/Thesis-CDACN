#pragma once
#include "Bounce.h"

class Level_1 : public Bounce
{
public:
	Level_1() : Bounce() 
	{
		obstacles.push_back(Wall(-0.7,0.5,0.3,0.1,-0.5,0.8,0.8,0.8));
		obstacles.push_back(Wall( 0.4,0.4,0.3,0.1, 0.5,0.8,0.8,0.8));
	}
};