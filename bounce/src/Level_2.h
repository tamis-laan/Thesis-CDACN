#pragma once
#include "Bounce.h"

class Level_2 : public Bounce
{
public:
	Level_2() : Bounce() 
	{
		obstacles.push_back(Wall(-0.6,0.5,0.5,0.1,-0.1,1.0,0.0,0.0));
		obstacles.push_back(Wall(0.1,0.5,0.5,0.1, -0.1,1.0,0.0,0.0));
		obstacles.push_back(Wall(-0.1,0.5,0.2,0.1, 0.1,0.0,1.0,0.0));
	}
};