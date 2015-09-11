//Include Logging
#include "Log.h"

//Include Levels
#include "src/Level_0.h"
#include "src/Level_1.h"
#include "src/Level_2.h"

//Include Agents
#include "Agents/Versions/DACN1.h"
#include "Agents/Versions/DACN2.h"
#include "Agents/Versions/DACN3.h"
#include "Agents/Versions/DACN4.h"
#include "Agents/Versions/CDACN1.h"
#include "Agents/Versions/CDACN2.h"
#include "Agents/Versions/CDACN3.h"
#include "Agents/Versions/CDACN4.h"

#include <iostream>
using namespace std;

//Interpolate between a and b in times t1 and t2
float interpolate(unsigned long t1, unsigned long t2, float a, float b, unsigned long t)
{
	float i = float(t)/(t2-t1);
	if(t<t1)
		return a;
	if(t>t2)
		return b;
	return b+(1.0-i)*(a-b);
}

//Run a experiment with a concolidated version of the System
template<typename SYSTEM, typename ENVIRONMENT> void experiment_c(string name, unsigned int length, unsigned int interval, unsigned int eval_length, bool disp = true)
{
	//Anounce start
	cout << "[!] " << name << endl;
	//Create environment
	ENVIRONMENT environment; 
	//Show environment
	if(disp) environment.show();
	//Create action descriptor
	ActionDescriptor descriptor({},{{-1.0f,1.0f}});
	//Create system
	SYSTEM system(descriptor,0.96,4,1000000,300,32,0.000025,0.000025,0.000025,1.0,1.0,512);
	//Go into experiment loop
	for(int i=1; i<length+1; i++)
		if(i%interval==0)
		{
			float avr_R = 0;
			float avr_Q = 0;
			system.exploration_rate(0.0);
			//Evaluate policy
			for(int j=0; j<eval_length; j++)
			{
				auto screen  = environment.screen();
				auto package = system.evaluate(screen);
				auto action  = package.first;
				auto reward  = environment.step(action);
				avr_R += reward;
				avr_Q += package.second;
			}
			//Log average reward and q values
			avr_Q = avr_Q/eval_length;
			avr_R = avr_R/eval_length;
			Log::instance().record("<R>",avr_R);
			Log::instance().record("<Q>",avr_Q);
			Log::instance().next();
			cout << "[*] " << float(i)/length*100.0 << "\%" << " expl. rate " << interpolate(0.0,length/4.0,1.0,0.0,i) << " <R> " << avr_R << " <Q> " << avr_Q << endl;
			//Reset the window
			system.reset();
			//Reset the environment
			environment.reset();
		}
		else
		{
			//Set exploration rate
			system.exploration_rate(interpolate(0.0,length/4.0,1.0,0.05,i));
			//Play
			auto screen = environment.screen();
			auto action = system.forward(screen).first;
			auto reward = environment.step(action);
			system.backward(reward,false);
			system.update();
			if(environment.key_code()==22)
				system.scan();
		}
	Log::instance().dump("/tmp/"+name+".csv");
}


//Run a experiment with a seperated version of the System
template<typename SYSTEM, typename ENVIRONMENT> void experiment_s(string name, unsigned int length, unsigned int interval, unsigned int eval_length, bool disp = true)
{
	//Anounce start
	cout << "[!] " << name << endl;
	//Create environment
	ENVIRONMENT environment; 
	//Show environment
	if(disp) environment.show();
	//Create action descriptor
	ActionDescriptor descriptor({},{{-1.0f,1.0f}});
	//Create system
	SYSTEM system(descriptor,0.96,4,1000000,300,32,0.000025,0.000025,1.0,1.0,512);
	//Go into experiment loop
	for(int i=1; i<length+1; i++)
		if(i%interval==0)
		{
			float avr_R = 0;
			float avr_Q = 0;
			system.exploration_rate(0.0);
			//Evaluate policy
			for(int j=0; j<eval_length; j++)
			{
				auto screen  = environment.screen();
				auto package = system.evaluate(screen);
				auto action  = package.first;
				auto reward  = environment.step(action);
				avr_R += reward;
				avr_Q += package.second;
			}
			//Log average reward and q values
			avr_Q = avr_Q/eval_length;
			avr_R = avr_R/eval_length;
			Log::instance().record("<R>",avr_R);
			Log::instance().record("<Q>",avr_Q);
			Log::instance().next();
			cout << "[*] " << float(i)/length*100.0 << "\%" << " expl. rate " << interpolate(0.0,length/4.0,1.0,0.0,i) << " <R> " << avr_R << " <Q> " << avr_Q << endl;
			//Reset the window
			system.reset();
			//Reset the environment
			environment.reset();
		}
		else
		{
			//Set exploration rate
			system.exploration_rate(interpolate(0.0,length/4.0,1.0,0.05,i));
			//Play
			auto screen = environment.screen();
			auto action = system.forward(screen).first;
			auto reward = environment.step(action);
			system.backward(reward,false);
			system.update();
			if(environment.key_code()==22)
				system.scan();
		}
	Log::instance().dump("/tmp/"+name+".csv");
}






















// void EXP_DACN3(unsigned int length = 2000, unsigned int interval = 1000, unsigned int eval_length = 500) 
// {
// 	cout << "[!] ACROBOT-DACN-3" << endl;
// 	Acrobot env;
// 	env.show();
// 	ActionDescriptor descriptor({},{{-1.0f,1.0f}});
// 	DACN3 system(descriptor,0.9,3,1000000,300,32,0.000025,0.000025,1.0,1.0,512);//0.000001
// 	for(int i=1; i<length+1; i++)
// 		if(i%interval==0)
// 		{
// 			float avr_R = 0;	float avr_Q = 0;
// 			system.exploration_rate(0.0);
// 			for(int j=0; j<eval_length; j++) //Play games for evaluation
// 			{
// 				auto screen  = env.screen();
// 				auto package = system.evaluate(screen);
// 				auto action  = package.first;
// 				auto reward  = env.step(action);
// 				avr_R += reward;
// 				avr_Q += package.second;
// 			}
// 			avr_Q = avr_Q/eval_length; avr_R = avr_R/eval_length;
// 			Log::instance().record("<R>",avr_R); Log::instance().record("<Q>",avr_Q); Log::instance().next();
// 			cout << "[*] " << float(i)/length*100.0 << "\%" << " ER " << interpolate(0.0,length/4.0,1.0,0.0,i) << " <R> " << avr_R << " <Q> " << avr_Q << endl;
// 			system.reset();
// 		}
// 		else //Run normally
// 		{
// 			system.exploration_rate(interpolate(0.0,length/4.0,1.0,0.0,i));
// 			auto screen = env.screen();
// 			auto action = system.forward(screen).first;
// 			auto reward = env.step(action);
// 			system.backward(reward,false);
// 			system.update();
// 			if(env.key_code()==22)
// 				system.scan();
// 		}
// 	Log::instance().dump("/tmp/ACROBOT-DACN-3.csv");
// }
