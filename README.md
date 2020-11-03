# Consolidated Deep Actor Critic Networks

#### Paper:
https://repository.tudelft.nl/islandora/object/uuid:682a56ed-8e21-4b70-af11-0e8e9e298fa2/datastream/OBJ/download 

#### Abstract:
The works [Volodymyr et al. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.] and [Volodymyr et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.] have demonstrated the power of combining deep neural networks with Watkins Q learning. They introduce deep Q networks (DQN) that learn to associate High dimensional inputs with Q values in order to produce discrete actions, allowing the system to learn complex strategies and play Atari games such as Breakout and Space invaders. Although powerful the system is limited to discrete actions. If we wish to control more complex systems like robots we need the ability to output multidimensional continuous actions. In this paper we investigate how to combine deep neural networks with actor critic models which have the ability to output multidimensional continuous actions. We name this class of systems deep actor critic networks (DACN) following the DQN naming convention. We derive and experiment with four methods to update the actor. We then consolidate the actor and critic networks into one unified network which we name consolidated deep actor critic networks (C-DACN). We hypothesize that consolidating the actor and critic networks might lead to faster convergence. We test the system in two environments named Acrobot (under actuated double pendulum) and Bounce (continuous action Atari Breakout look alike).

#### Folders:
`/acrobot`: Double pendulum robot simulator
`/bounce`: Simplified breakout game simulator 
`/ddnet`: Custom deep learning framework in C++/CUDA (written before tensorflow/pytorch existed)
`/ilm`: Reinforcement learning framework and experiments
