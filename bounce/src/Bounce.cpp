#include "Bounce.h"

//Constructor
Bounce::Bounce() : Environment(84, 84, 336, 336, true), left(-11.0,-1,10.1,12.0), right(0.9,-1,10.0,12.0), top(-0.9,0.9,1.8,10.0)
{
	reset();
}

//Reset
void Bounce::reset()
{
	std::uniform_real_distribution<> unifrom_x(-0.01,0.01);
	std::uniform_real_distribution<> unifrom_y(-0.04,-0.02);
	ball.position.x = 0.0;
	ball.position.y = 0.4;
	ball.velocity.x = unifrom_x(engine);
	ball.velocity.y = unifrom_y(engine);

	//Normalize velocity
	float n = sqrt(pow(ball.velocity.x,2)+pow(ball.velocity.y,2));
	ball.velocity.x = 0.03*ball.velocity.x/n;
	ball.velocity.y = 0.03*ball.velocity.y/n;
}

//Compute game logic
float Bounce::logic(unsigned int steps)
{
	//Set default reward to 0
	float reward = 0.0;
	//Step logic
	for(int step = 0; step<steps; step++)
	{
		//Static wall logic
		reward += top.logic(ball);
		reward += right.logic(ball);
		reward += left.logic(ball);
		//Dynamic wall logic
		for(int i=0; i<obstacles.size(); i++)
			reward += obstacles[i].logic(ball);
		//Paddle logic
		reward += paddle.logic(ball);
		//Ball logic
		reward += ball.logic();
		//Check end game
		if(ball.position.y <= -1.0)
		{
			reset();
			reward += -1.0;
		}
	}
	//Return reward
	return reward;
}

//Draw the game
void Bounce::draw()
{
	//Clear scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//Draw paddle
	paddle.draw();
	//Draw ball
	ball.draw();
	//Draw left wall
	left.draw();
	//Draw right wall
	right.draw();
	//Draw top wall
	top.draw();
	//Draw obstacles
	for(int i=0; i<obstacles.size(); i++)
		obstacles[i].draw();
}

float Bounce::step(std::vector<float> action)
{
	//Move paddle
	paddle.move(action[0]);
	//Execute logic
	float reward = logic();
	score += reward;
	//Draw
	draw();
	// Render
	render();
	// glutSwapBuffers();
	//Return reward
	return 2.25*reward;
}	