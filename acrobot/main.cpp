#include <iostream>
#include "experiments.cpp"

//Run experiments
void run()
{
	//Concolidated version
	experiment_c<CDACN4,Acrobot>("ACROBOT-CDACN-H",400000,2000,2000,true);
	experiment_c<CDACN3,Acrobot>("ACROBOT-CDACN-G",400000,2000,2000,true);
	experiment_c<CDACN1,Acrobot>("ACROBOT-CDACN-S",400000,2000,2000,true);
	experiment_c<CDACN2,Acrobot>("ACROBOT-CDACN-TD",400000,2000,2000,true);
	//Seperated version
	experiment_s<DACN4,Acrobot>("ACROBOT-DACN-H",400000,2000,2000,true);
	experiment_s<DACN3,Acrobot>("ACROBOT-DACN-G",400000,2000,2000,true);
	experiment_s<DACN1,Acrobot>("ACROBOT-DACN-S",400000,2000,2000,true);
	experiment_s<DACN2,Acrobot>("ACROBOT-DACN-TD",400000,2000,2000,true);
}

int main(int argc, char *argv[])
{
	experiment_s<DACN4,Acrobot>("ACROBOT-DACN-H",400000,2000,2000,true);
	experiment_s<DACN3,Acrobot>("ACROBOT-DACN-G",400000,2000,2000,true);
	experiment_s<DACN1,Acrobot>("ACROBOT-DACN-S",400000,2000,2000,true);
	experiment_s<DACN2,Acrobot>("ACROBOT-DACN-TD",400000,2000,2000,true);
	// experiment_c<CDACN1,Acrobot>("ACROBOT-CDACN-S",400000,2000,2000,true);
	// run();
	// experiment_c<CDACN1,Acrobot>("ACROBOT-CDACN-S",400000,2000,2000,true);
	// experiment_s<DACN1,Acrobot>("ACROBOT-DACN-S",400000,2000,2000,true);
	// experiment_s<DACN2,Acrobot>("ACROBOT-DACN-TD",400000,2000,2000,true);

	// Acrobot env;
	// ActionDescriptor descriptor({},{{-1.0f,1.0f}});
	// CDACN4 system(descriptor,0.9,3,1000000,300,32,0.000025,0.000025,0.000025,1.0,1.0,128);
	// env.show();
	// for(int i=0; i<1000000000; i++)
	// {
	// 	system.exploration_rate(interpolate(0,40000,1.0,0.0,i));
	// 	auto screen  = env.screen();
	// 	auto package = system.forward(screen);
	// 	auto action  = package.first;
	// 	auto reward  = env.step(action);
	// 	system.backward(reward,false);
	// 	system.update();
	// 	if(env.key_code()==22)
	// 			system.scan();
	// 	cout << "Epoch " << i << " Reward: " << reward << " Critic: " << package.second << " Actor: " << package.first[0] << endl;
	// }
}