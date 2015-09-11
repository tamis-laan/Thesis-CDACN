#include "Objects.h"
#include <GL/glut.h>
#include <algorithm>
//Draw rectangle
void rectangle(float x, float y, float w, float h, float r = 1.0, float g = 1.0, float b = 1.0)
{
	glBegin(GL_QUADS);
		glColor3f(r, g, b);
		glVertex2f(x,   y  );
		glVertex2f(x,   y+h);
		glVertex2f(x+w, y+h);
		glVertex2f(x+w, y  );
	glEnd();
}

//Draw the ball on screen
void  Ball::draw() 
{
	rectangle(position.x-size/2.0, position.y-size/2.0, size, size, color.r, color.g, color.b);
}

//Evaluate ball logic
float Ball::logic()
{
	//Maximum velocity?
	// velocity.x = max(velocity.x,1.0);
	// velocity.y = max(velocity.y,1.0);
	//Ball dynamics
	position.x = position.x+velocity.x;
	position.y = position.y+velocity.y;
	return 0.0;
}

void Paddle::draw() 
{
	rectangle(position_1 - size.x/2.0f, -1, size.x, size.y, color.r, color.g, color.b);
}

void Paddle::move(float a)
{
	//Clip between -1 and 1
	a = std::max(-1.0f,std::min(1.0f,a));
	//Scale down
	a = a/7.5f;
	//Save old paddle position
	position_0 = position_1;
	//Set new position
	position_1 = std::max(-0.9f+size.x/2.0f,std::min(0.9f-size.x/2.0f,position_1+a));
}

float Paddle::logic(Ball &ball)
{
	//Paddle dynamics
	if(ball.position.y+ball.velocity.y+ball.size < -1+size.y+ball.size)
		if(ball.position.x+ball.velocity.x <= position_1+size.x/2.0+ball.size/2.0 and ball.position.x+ball.velocity.x >= position_1-size.x/2.0-ball.size/2.0 )
		{
			//Threshold reflect ball
				// float diff = ball.position.x-position_1;
				// ball.velocity.y  = -ball.velocity.y;
				// ball.velocity.x +=  (diff>0)*0.01 + (diff<0)*-0.01;

			//Reflect ball based on position
				// float diff = ball.position.x-position_1;
				// ball.velocity.y  = -ball.velocity.y;
				// ball.velocity.x +=  diff*0.15;
			
			//Reflect ball based on friction
				ball.velocity.y  = -ball.velocity.y; 
				ball.velocity.x += (position_1-position_0)*0.15;
		}
	return 0.0;	
}

Wall::Wall(float x, float y, float w, float h, float reward, float r, float g, float b) : reward(reward)
{
	position.x = x; 
	position.y = y;
	size.x     = w;
	size.y     = h;
	color.r    = r;
	color.g    = g; 
	color.b    = b;
}

void Wall::draw()
{
	rectangle(position.x, position.y, size.x, size.y, color.r, color.g, color.b);
}

float Wall::logic(Ball &ball)
{
	//old position
	float b0x = ball.position.x;
	float b0y = ball.position.y;
	//new position
	float b1x = ball.position.x+ball.velocity.x;
	float b1y = ball.position.y+ball.velocity.y;

	//Check if ball is inside wall in the next time step. 
		if(b1x+ball.size/2.0>=position.x and b1x-ball.size/2.0<=position.x+size.x and b1y+ball.size/2.0>=position.y and b1y-ball.size/2.0<=position.y+size.y)
		{
			//Check bottom
			if(b0y<=position.y)
				ball.velocity.y = -ball.velocity.y;
			//Check top
			else if(b0y>=position.y+size.y)
				ball.velocity.y = -ball.velocity.y;
			//Check left
			else if(b0x<=position.x)
				ball.velocity.x = -ball.velocity.x;
			//Check right
			else if(b0x>=position.x+size.x)
				ball.velocity.x = -ball.velocity.x;
			//Otherwise corner case
			else
			{
				ball.velocity.y = -ball.velocity.y;
				ball.velocity.x = -ball.velocity.x;
			}
			return reward;
		}
		return 0;
}