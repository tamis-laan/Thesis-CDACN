#include <iostream>

#include "src/Agents/DACN.h"

#include <ale/ale_interface.hpp>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_dynamic_io.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/gil/extension/numeric/resample.hpp>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

using namespace std;
using namespace boost::gil;

//Load the image and crop/rescale it to 84x84
gray8_image_t to_image(ALEScreen &screen)
{
	gray8_image_t img(160, 160);
	gray8_image_t::view_t v = view(img);
	for(int x=0;x<160;x++)
		for(int y=0;y<160;y++)
			v(y,x) = gray8_pixel_t( static_cast<unsigned char>(screen.get(x+40,y)) );
	gray8_image_t rimg(84, 84);
	resize_view(const_view(img), view(rimg), bilinear_sampler());
	// png_write_view("screen.png", const_view(rimg));
	return rimg;
}

// Vectorize the image and normalize input between 0 and 1
vector<unsigned char> to_input(gray8_image_t &img)
{
	gray8_image_t::view_t v = view(img);
	vector<unsigned char> input(v.width()*v.height());
    for (int y=0; y<v.height(); y++)
        for (int x=0; x<v.width(); x++)
        	input[x*v.width()+y] = v(x,y);
    return input;
}

//Run the Arcade Learning Environment using the DQN agent.
void run_ale(int argc, char** argv)
{
	//Create Arcade Learning Environment
	ALEInterface* ale = new ALEInterface(false);
	//Load the Atari Rom we are going to play
	ale->loadROM(argv[1]);
	//Get the set of possible actions from ALE ROM
	ActionVect action_set = {PLAYER_A_LEFTFIRE,PLAYER_A_FIRE,PLAYER_A_RIGHTFIRE}; //ale->getMinimalActionSet();
	//Create action descriptor
	ActionDescriptor descriptor({action_set.size()},{});
	//Create Learning System
	DACN system(descriptor,0.9,4,1000000,100,32);
	//Set exploration rate
	system.exploration_rate(0.9);

	cudaProfilerStart(); nvtxRangePushA("2 Atari Games");
	for(int episode=0; episode<2; episode++)
	{	
		string tmp = "episode: " + to_string(episode);
		nvtxRangePushA(tmp.c_str());

		//Restart the game
		ale->reset_game();

		//Game Loop
		while(!ale->game_over())
		{
			nvtxRangePushA("step");
				//Convert screen to input
				ALEScreen screen = ale->getScreen();
				gray8_image_t img = to_image(screen);
				vector<unsigned char> input = to_input(img);
				float raw_action = system.forward( input )[0];
				//cast action
				int action = static_cast<int>(raw_action);
				//Execute the action and get the reward
				float reward = ale->act(action_set[action]);
				//Normalize the reward
				float normalized_reward = max(min(1.0f,reward),-1.0f);
				//Backward the result
				system.backward(normalized_reward,ale->game_over());
			nvtxRangePop();
		}
		nvtxRangePop();
	}
	cudaProfilerStop(); nvtxRangePop();
}

//Main function
int main(int argc, char** argv)
{
	run_ale(argc,argv);
}