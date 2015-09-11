#include <iostream>
#include "experiments.cpp"

//Run experiments
void run()
{
	//Concolidated version
	experiment_c<CDACN4,Level_1>("BOUNCE-CDACN-H",400000,2000,2000,true);
	experiment_c<CDACN3,Level_1>("BOUNCE-CDACN-G",400000,2000,2000,true);
	experiment_c<CDACN1,Level_1>("BOUNCE-CDACN-S",400000,2000,2000,true);
	experiment_c<CDACN2,Level_1>("BOUNCE-CDACN-TD",400000,2000,2000,true);
	//Seperated version
	experiment_s<DACN4,Level_1>("BOUNCE-DACN-H",400000,2000,2000,true);
	experiment_s<DACN3,Level_1>("BOUNCE-DACN-G",400000,2000,2000,true);
	experiment_s<DACN1,Level_1>("BOUNCE-DACN-S",400000,2000,2000,true);
	experiment_s<DACN2,Level_1>("BOUNCE-DACN-TD",400000,2000,2000,true);
}

int main(int argc, char *argv[])
{
	experiment_s<DACN2,Level_1>("BOUNCE-DACN-TD",400000,2000,2000,true);

	// Level_1 env;
	// ActionDescriptor descriptor({},{{-1.0f,1.0f}});
	// CDACN4 system(descriptor,0.96,3,1000000,300,32,0.000025,0.000025,0.000025,1.0,1.0,128);//0.000001
	// env.show();
	// for(int i=0; i<100000; i++)
	// {
	// 	system.exploration_rate(interpolate(0,50000,1.0,0.1,i));
	// 	auto screen  = env.screen();
	// 	auto package = system.forward(screen);
	// 	auto action  = package.first;
	// 	auto reward  = env.step(action);
	// 	system.backward(reward,false);
	// 	system.update();
	// 	if(env.key_code()==22)
	// 		system.scan(); 	
	// 	cout << "Critic: " << package.second << " Actor: " << package.first[0] << endl;
	// }
}