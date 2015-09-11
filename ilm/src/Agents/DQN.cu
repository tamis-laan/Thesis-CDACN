// #include "DQN.h"
// #include <thrust/extrema.h>

// //Constructor
// DQN::DQN(ActionDescriptor descriptor, float discount, int temporal_stride, int capacity, int burn_in, int batch_size) : 
// 			Agent(descriptor,temporal_stride,84,84,capacity,burn_in,batch_size),
// 			discount_(discount),
// 			exploration_rate_(1.0),
// 			uniform_discrete(0,descriptor.discrete_actions_num(0)-1), 
// 			uniform_real(0.0,1.0)
// {
// 	//Create network device handle
// 	cudnnCreate(&handle);
// 	//Create the actual network
// 	network = DQN_N(handle,{1,batch_size,batch_size},unique_ptr<ddnet::ADAM>(new ddnet::ADAM(0.00025,0.95,0.95,1e-10)),temporal_stride,descriptor.discrete_actions_num(0));
// }

// //Executes explore step
// vector<float> DQN::explore()
// { 
// 	return { static_cast<float>(uniform_discrete(engine)) }; 
// }

// //Executes exploit step
// vector<float> DQN::exploit(const Window &window)
// {
// 	//Extract the right state from the window
// 	vector<float> state = window.states_right();
// 	//Copy the right state into the NN input tentsor
// 	thrust::copy(state.begin(),state.end(),network.input(0).data(0).begin());
// 	//Forward the data to compute the output
// 	network.forward(0);
// 	//Get the output of the NN as a Tensor variable
// 	auto output = network.output(0).data(0);
// 	//Find the maximum output value i.e. maximum Q value
// 	auto iter = thrust::max_element( output.begin(), output.end() );
// 	//Get the action
// 	int action = iter-output.begin();
// 	// Print results
// 	cout << "Action: " << action << " Q value: " << output[action] << endl;
// 	//Return the index of the maximum Q value
// 	return { static_cast<float>(action) };
// }

// //This method executes the policy
// vector<float> DQN::policy(const Window &window)
// {
// 	if(uniform_real(engine)<exploration_rate_ or not window.full())
// 		return explore();
// 	else
// 		return exploit(window);
// }

// //This method executes one learning step
// void DQN::step(const vector<Window> &batch)
// {
// 	//Allocate 
// 	vector<float> lbuffer(0);
// 	vector<float> rbuffer(0);
// 	lbuffer.reserve(width_*height_*temporal_stride_*batch_size_);
// 	rbuffer.reserve(width_*height_*temporal_stride_*batch_size_);
// 	for(int i=0; i<batch_size_; i++)
// 	{
// 		auto l = batch[i].states_left();
// 		lbuffer.insert(lbuffer.end(),l.begin(),l.end());
// 		auto r = batch[i].states_right();
// 		rbuffer.insert(rbuffer.end(),r.begin(),r.end());
// 	}
// 	//Upload states to device
// 	thrust::copy(lbuffer.begin(),lbuffer.end(),network.input(0).data(1).begin());
// 	thrust::copy(rbuffer.begin(),rbuffer.end(),network.input(0).data(2).begin());

// 	//Forward Q(s1,a) and Q(s2,a) through the nework 
// 	network.forward(1);
// 	network.forward(2);

// 	//Compute new Q values (  reward + discount_*max_a[Q(s2,a)]  )
// 	vector<float> left(descriptor_.discrete_actions_num(0));
// 	vector<float> right(descriptor_.discrete_actions_num(0));
// 	vector<float> delta(descriptor_.discrete_actions_num(0));

// 	for(int i=0; i<batch_size_; i++)
// 	{	
// 		//Copy from device to host
// 		thrust::copy( network.output(0).data(1,i), network.output(0).data(1,i)+descriptor_.discrete_actions_num(0), left.begin());
// 		thrust::copy( network.output(0).data(2,i), network.output(0).data(2,i)+descriptor_.discrete_actions_num(0), right.begin());
// 		//Get the maximum q value
// 		float m = *max_element(right.begin(),right.end());
// 		//Get the action 
// 		int   a = batch[i].actions_left()[temporal_stride_-1][0];
// 		//Get the reward 
// 		float r = batch[i].rewards_left()[temporal_stride_-1];
// 		//Get the terminal
// 		int   t = batch[i].terminals_left()[temporal_stride_-1];
// 		//Zero out the delta values
// 		fill(delta.begin(),delta.end(),0.0);
// 		//Set the correct new left Q value delta
// 		delta[a] = r+discount_*m*(1.0-t) - left[a];
// 		//Copy delta host to device
// 		thrust::copy(delta.begin(),delta.end(), network.output(0).delta(1,i) );
// 	}
// 	//Backward (train network) the new values
// 	// network.backward_update(1);
// 	counter = (counter+1)%update_frequency;
// 	network.backward_update(1, counter!=0 );
// }

// // DQN Network specification Constructor. Constructs the layout of the network.
// DQN_N::DQN_N(cudnnHandle_t handle, vector<int> batch, unique_ptr<ddnet::Solver> solver, int temporal_stride, int number_of_actions) : ddnet::Network(handle,batch,move(solver))
// {
// 	//Refrence myself
// 	Network &net = *this;
// 	//Create input block 84x84 Atari image with 1 channel (Black&White)
// 	auto b1 = ddnet::create_block(net,temporal_stride,84,84);
// 	//Create first conv layer
// 	auto l1 = ddnet::create_element<ddnet::ConvolutionLayer>(net,32,8,8,4,4,0.01); //2.0/(temporal_stride*84*84)
// 	//Create intermediate block
// 	auto b2 = ddnet::create_block(net,32,20,20);
// 	//Create Activation Layer
// 	auto l2 = ddnet::create_element<ddnet::ActivationLayer>(net,CUDNN_ACTIVATION_RELU);

// 	//Create intermediate block
// 	auto b3 = ddnet::create_block(net,32,20,20);
// 	//Create second conv layer
// 	auto l3 = ddnet::create_element<ddnet::ConvolutionLayer>(net,64,4,4,2,2,0.01); //2.0/(32*20*20)
// 	//Create intermediate block
// 	auto b4 = ddnet::create_block(net,64,9,9);
// 	//Create Activation Layer
// 	auto l4 = ddnet::create_element<ddnet::ActivationLayer>(net,CUDNN_ACTIVATION_RELU);

// 	//Create intermediate block
// 	auto b5 = ddnet::create_block(net,64,9,9);
// 	//Create third conv layer
// 	auto l5 = ddnet::create_element<ddnet::ConvolutionLayer>(net,64,3,3,1,1,0.01); //2.0/(64*9*9)
// 	//Create intermediate block
// 	auto b6 = ddnet::create_block(net,64,7,7);
// 	//Create Activation Layer
// 	auto l6 = ddnet::create_element<ddnet::ActivationLayer>(net,CUDNN_ACTIVATION_RELU);

// 	//Create intermediate block
// 	auto b7 = ddnet::create_block(net,64,7,7);
// 	//Create denseLayer
// 	auto l7 = ddnet::create_element<ddnet::DenseLayer>(net,512,0.01); //2.0/(64*7*7)
// 	//Create intermediate block
// 	auto b8 = ddnet::create_block(net,512);
// 	//Create Activation Layer
// 	auto l8 = ddnet::create_element<ddnet::ActivationLayer>(net,CUDNN_ACTIVATION_RELU);
// 	//Create intermediate block
// 	auto b9 = ddnet::create_block(net,512);

// 	//Create denseLayer
// 	auto l9 = ddnet::create_element<ddnet::DenseLayer>(net,number_of_actions,0.01); //1.0/512.0
// 	//Create intermediate block
// 	auto b10 = ddnet::create_block(net,number_of_actions);

// 	//Link layers and blocks together
// 	ddnet::link(b1,l1,b2);
// 	ddnet::link(b2,l2,b3);
// 	ddnet::link(b3,l3,b4); 
// 	ddnet::link(b4,l4,b5);
// 	ddnet::link(b5,l5,b6);
// 	ddnet::link(b6,l6,b7);
// 	ddnet::link(b7,l7,b8);
// 	ddnet::link(b8,l8,b9);
// 	ddnet::link(b9,l9,b10);

// 	//Set input/output blocks
// 	ddnet::add_input(net,b1);
// 	ddnet::add_output(net,b10);

// 	//Set execution order
// 	ddnet::push(net,b1);
// 	ddnet::push(net,l1);
// 	ddnet::push(net,b2);
// 	ddnet::push(net,l2);
// 	ddnet::push(net,b3);
// 	ddnet::push(net,l3);
// 	ddnet::push(net,b4);
// 	ddnet::push(net,l4);
// 	ddnet::push(net,b5);
// 	ddnet::push(net,l5);
// 	ddnet::push(net,b6);
// 	ddnet::push(net,l6);
// 	ddnet::push(net,b7);
// 	ddnet::push(net,l7);
// 	ddnet::push(net,b8);
// 	ddnet::push(net,l8);
// 	ddnet::push(net,b9);
// 	ddnet::push(net,l9);
// 	ddnet::push(net,b10);
// }