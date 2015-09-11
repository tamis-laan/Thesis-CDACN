#include "Environment.h"
#include <iostream>
#include <cmath>

#define PI 3.14159265359f

class Acrobot : public Environment
{
private:
	//Constants
	float m1 = 1;		//Mass 1
	float m2 = 1;		//Mass 2
	float l1 = 0.45;	//Link length 1
	float l2 = 0.45;		//Link length 2
	float g  = 9.8;		//Gravitation Constant
	float I1 = 1.0;		//Moment of inertia 1
	float I2 = 1.0;		//Moment of inertia 2

	//State
	float t1 =  0.0;
	float t2 =  0.0;
	float dt1 = 0.0;
	float dt2 = 0.0;

	//inputs
	float u = 0.0;

	//Reward
	float reward = 0;

public:
	Acrobot() : Environment(84, 84, 336, 336, false) {} 

	//Draw circle
	void circle(float x, float y, float radius = 0.075, float r = 1.0, float g = 1.0, float b = 1.0)
	{
		int segments = 400;
		glColor3f(r, g, b);
		glBegin( GL_TRIANGLE_FAN );
		glVertex2f(x, y);
		for( int n = 0; n <= segments; ++n ) 
		{
			float const t = 2*PI*(float)n/(float)segments;
			glVertex2f(x + sin(t)*radius, y + cos(t)*radius);
		}
		glEnd();
	}

	//Draw line
	void line(float x1, float y1, float x2, float y2, float w = 1.0, float r = 1.0, float g = 1.0, float b = 1.0)
	{
		glLineWidth(w); 
		glColor3f(r, g, b);
		glBegin(GL_LINES);
		glVertex2f(x1, y1);
		glVertex2f(x2, y2);
		glEnd();
	}

	void solver(float h, int epochs)
	{
		float lc1 = l1/2.0;
		float lc2 = l2/2.0;
		for(int i=0; i<epochs; i++)
		{
			//Compute acceleration
			float d1    = m1*pow(lc1,2)+m2*(pow(l1,2)+pow(lc2,2)+2*l1*lc2*cos(t2)+I1+I2);
			float d2    = m2*(pow(lc2,2)+l1*lc2*cos(t2))+I2;
			float phi_2 = m2*lc2*g*cos(t1+t2-PI/2.0);
			float phi_1 = -m2*l1*lc2*pow(dt2,2)*sin(t2)-2*m2*l1*lc2*dt2*dt1*sin(t2)+(m1*lc1+m2*l1)*g*cos(t1-PI/2.0)+phi_2;

			float ddt2 = 1.0/(m2*pow(lc2,2)+I2-pow(d2,2)/d1)*(u+d2/d1*phi_1-m2*l1*lc2*pow(dt1,2)*sin(t2)-phi_2);
			float ddt1 = -1.0/d1*(d2*ddt2+phi_1);

			//Simulate forward
			t1  = fmod(t1+h*dt1,2*PI);
			t2  = fmod(t2+h*dt2,2*PI);
			dt1 = std::min(std::max(dt1+h*ddt1,-5.0f),5.0f);
			dt2 = std::min(std::max(dt2+h*ddt2,-5.0f),5.0f);

		}
	}

	virtual float step(std::vector<float> action)
	{
		//Set action
			u = action[0]*5;
		//Do a simulation step
			solver(0.01,15);
		//Move to cartesian coordinates
			float x1 =  l1*sin(t1);
			float y1 = -l1*cos(t1);
			float x2 = x1 + l2*sin(t1+t2);
			float y2 = y1 - l2*cos(t1+t2);
		//Draw the result
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			circle(x1 ,y1 ,0.06,1.0,1.0,1.0);
			circle(x2 ,y2 ,0.06,1.0,1.0,1.0);
			circle(0.0,0.0,0.06,1.0,1.0,1.0);
			line(0.0,0.0,x1,y1,1.0,0.5,0.5,0.5);
			line(x1 ,y1 ,x2,y2,1.0,0.5,0.5,0.5);
			render();
		//Return the reward signal
			return 0.1*(  (y1/l1 + (y2-y1)/l2)/2.0  );
	}


};