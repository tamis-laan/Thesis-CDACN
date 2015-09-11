#include "Window.h" 
/*
	The compressor, as it's name suggests, compresses data. In particular it compresses 84x84 atari screen images. 
	Compression makes it feasible to store many more images in RAM without using disk space. The lz4 compression algorithm
	is reasonably fast for compression, but ridiculously fast for decompression. We use significantly more decompression then 
	compression which makes this compression algorithm perfect.
*/
namespace compressor
{
	//Compressor Buffer size 
	#define COMP_BUFF_SIZE   84*84*5
	//Decompressor Buffer size 
	#define DECOMP_BUFF_SIZE 84*84*5
	//Compressor buffer
	static unsigned char* compression_buffer   = new unsigned char[COMP_BUFF_SIZE];
	//Decompressor buffer
	static unsigned char* decompression_buffer = new unsigned char[DECOMP_BUFF_SIZE];
	//Compress data 
	static vector<unsigned char> compress_lz4(vector<unsigned char> input)
	{
			const char* rc1 = reinterpret_cast<const char*>( input.data() );
			char* rc2 = reinterpret_cast<char*>( compression_buffer );
			int dst_len = LZ4_compress(rc1, rc2, input.size());
			vector<unsigned char> output(dst_len);
			output.assign(compression_buffer,compression_buffer+dst_len);
			return move(output);
	}
	//Decompress data
	static vector<unsigned char> decompress_lz4(vector<unsigned char> input)
	{
			const char* rc1 = reinterpret_cast<const char*>( input.data() );
			char* rc2 = reinterpret_cast<char*>( decompression_buffer );
			int dst_len = LZ4_decompress_safe(rc1, rc2, input.size(), DECOMP_BUFF_SIZE);
			vector<unsigned char> output(dst_len);
			output.assign(decompression_buffer,decompression_buffer+dst_len);
			return move(output);
	}
}

//Constructor
Window::Window(int temporal_stride, int state_width, int state_height, int action_dim) : 
		temporal_stride_(temporal_stride), 
		state_width_(state_width), 
		state_height_(state_height), 
		action_dim_(action_dim),
		states_(temporal_stride+1), 
		actions_(temporal_stride+1), 
		rewards_(temporal_stride+1), 
		terminals_(temporal_stride+1)
		{
			//Allocate and compress for state stub 
			state_stub_ = compressor::compress_lz4(vector<unsigned char>(state_width_*state_height_,'x'));
		}

//This method pushes a null state (using stub) in order to mark end of episode
void Window::push_stub()
{
	//Push the states_ stub
	states_.push_back(state_stub_);
	//Push the actions_ stub
	actions_.push_back(vector<float>(action_dim_,-1337.0));
	//Push the rewards stub
	rewards_.push_back(-1337.0);
	//Push the terminals stub
	terminals_.push_back(-1337);
}

//Push a new state into the window
void Window::push_state(const vector<unsigned char> &state)
{
	states_.push_back(compressor::compress_lz4(state));
}

//Push a new action
void Window::push_action(const vector<float> &action)
{
	actions_.push_back(action);
}

//Push a new reward
void Window::push_reward(float reward)
{
	rewards_.push_back(reward);
}

//Push a new terminal
void Window::push_terminal(int terminal)
{
	terminals_.push_back(terminal);
}

//Resets the states, actions, rewards and terminals 
void Window::reset()
{
	states_.clear();
	actions_.clear();
	rewards_.clear();
	terminals_.clear();
}

//Return the number of pushed elements in the window
int Window::size() const
{
	return states_.size();
}

//Return the capacity of the window
int Window::capacity() const
{
	return temporal_stride_+1;
}

//Check if the window is full
bool Window::full() const 
{
	return states_.size() == temporal_stride_+1;
}

//Return the left state
vector<float> Window::states_left() const
{
	vector<float> buffer;
	buffer.reserve(state_width_*state_height_*temporal_stride_);
	for(int i=0; i<temporal_stride_;i++)
	{
		vector<unsigned char> t = compressor::decompress_lz4(states_[i]);
		buffer.insert(buffer.end(),t.begin(),t.end());
	}
	transform(buffer.begin(),buffer.end(),buffer.begin(),[](float x){return x/265.0;});
	return move(buffer);
}


//Return the right states
vector<float> Window::states_right() const
{
	vector<float> buffer;
	buffer.reserve(state_width_*state_height_*temporal_stride_);
	for(int i=0; i<temporal_stride_;i++)
	{
		vector<unsigned char> t = compressor::decompress_lz4(states_[i+1]);
		buffer.insert(buffer.end(),t.begin(),t.end());
	}
	transform(buffer.begin(),buffer.end(),buffer.begin(),[](float x){return x/265.0;});
	return move(buffer);
}

//Return the left actions
vector<vector<float>> Window::actions_left() const
{
	vector<vector<float>> buffer(temporal_stride_);
	buffer.assign(actions_.begin(),actions_.end()-1);
	return move(buffer);
}

//Return the right actions
vector<vector<float>> Window::actions_right() const
{
	vector<vector<float>> buffer(temporal_stride_);
	buffer.assign(actions_.begin()+1,actions_.end());
	return move(buffer);
}

//Return left rewards
vector<float> Window::rewards_left() const
{
	vector<float> buffer(temporal_stride_);
	buffer.assign(rewards_.begin(),rewards_.end()-1);
	return move(buffer);
}

//Return right rewards
vector<float> Window::rewards_right() const
{
	vector<float> buffer(temporal_stride_);
	buffer.assign(rewards_.begin()+1,rewards_.end());
	return move(buffer);
}

//Return left terminals
vector<int> Window::terminals_left() const 
{
	vector<int> buffer(temporal_stride_);
	buffer.assign(terminals_.begin(),terminals_.end()-1);
	return move(buffer);
}

//Return right terminals
vector<int> Window::terminals_right() const 
{
	vector<int> buffer(temporal_stride_);
	buffer.assign(terminals_.begin()+1,terminals_.end());
	return move(buffer);
}