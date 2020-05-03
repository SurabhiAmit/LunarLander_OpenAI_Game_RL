Project description:

Project develops intelligent agents using deep reinforcement learning techniques of Deep Q-Network(DQN) and Double deep Q-Network(DDQN) to play the LunarLander game. DQN combines Q-learning with a deep neural network,by using neural networks as a policy and use "hacks" like experience replay, target networks and reward clipping.

The difference between DQN and DDQN is in the calculation of the target Q-values of the next states. In DQN, we simply take the maximum of all the Q-values over all possible actions. This is likely to select over-estimated values, hence DDPG(deep deterministic policy gradient) proposed to estimate the value of the chosen action instead. The chosen action is the one selected by our policy model. In DQN, the target Q-Network selects and evaluates every action resulting in an overestimation of Q value. To resolve this issue, DDQN proposes to use the Q-Network (policy model) to choose the action (Action Selection) and use the target Q-Network to evaluate the action (Action Evaluations).

The agent is trained for 1500 training episodes and tested for next 100 episodes. Hyperparameter tuning with respect to learning rate, discount factor, tau and epsilon decay are included in the project code. The plots showing reward per episode and moving average of rewards for both training and tetsing phase also get generated.

How to run the source code?

project2.py

1. Open project2.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where project2.py is stored and type "project2.py" and hit Enter key.
3. The graph showing the rewards per episode and moving average of rewards over a window of 100 days, for the training period, gets generated.
4. On closing this graph window, the graph showing the rewards per episode and moving average of rewards over a window of 25 days, for the testing period, gets generated.

tune_alpha_vectorised.py
1. Open tune_alpha_vectorised.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where tune_alpha_vectorised.py is stored and type "tune_alpha_vectorised.py" and hit Enter key.
3. The graph showing the moving average of rewards over a window of 100 episodes for alpha values 0.00005, 0.00025 and 0.0005, during the training period, gets generated.
4. On closing this graph window, the graph showing the average reward for 100 test episodes for the alpha values 0.00005, 0.00025 and 0.0005 gets generated.

tune_gamma_vectorised.py
1. Open tune_gamma_vectorised.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where tune_gamma_vectorised.py is stored and type "tune_gamma_vectorised.py" and hit Enter key.
3. The graph showing the moving average of rewards over a window of 100 episodes for the gamma values 1,0.99 and 0.79, during the training period, gets generated.
4. On closing this graph window, the graph showing the average reward for 100 test episodes for the gamma values 1, 0.99 and 0.79, gets generated.

tune_decay_vectorised.py
1. Open tune_decay_vectorised.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where tune_decay_vectorised.py is stored and type "tune_decay_vectorised.py" and hit Enter key.
3. The graph showing the moving average of rewards over a window of 100 episodes for the epsilon-decay rate values 0.999, 0.995 and 0.8, during the training period, gets generated.
4. On closing this graph window, the graph showing the average reward for 100 test episodes for the epsilon-decay rate values 0.999, 0.995 and 0.8, gets generated.

tune_tau_vectorised.py
1. Open tune_tau_vectorised.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where tune_tau_vectorised.py is stored and type "tune_tau_vectorised.py" and hit Enter key.
3. The graph showing the moving average of rewards over a window of 100 episodes for the tau values 0.0001,0.001 and 0.005, during the training period, gets generated.
4. On closing this graph window, the graph showing the average reward for 100 test episodes for the tau values 0.0001,0.001 and 0.005, gets generated.

project_2_with_saved_weights.py
1. Open project2_with_saved_weights.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where project2_with_saved_weights.py is stored and type "project2_with_saved_weights.py" and hit Enter key.
3. The graph showing the rewards per episode and the average reward for 100 episodes gets generated, using the model that is loaded with weights saved in "model_weights2.h5".

project2_DQN_two_networks.py
1. Open project2_DQN_two_networks.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where project2_DQN_two_networks.py is stored and type "project2_DQN_two_networks.py" and hit Enter key.
3. The graph showing the rewards per episode and moving average of rewards over a window of 100 days, for the training period, gets generated.
4. On closing this graph window, the graph showing the rewards per episode and moving average of rewards over a window of 25 days, for the testing period, gets generated.

project2_DQN_one_network.py
1. Open project2_DQN_one_network.py in IDE like PyCharm and click 'Run'.
2. Or, on command prompt, go to the location where project2_DQN_one_network.py is stored and type "project2_DQN_one_network.py" and hit Enter key.
3. The graph showing the rewards per episode and moving average of rewards over a window of 100 days, for the training period, gets generated.
4. On closing this graph window, the graph showing the rewards per episode and moving average of rewards over a window of 25 days, for the testing period, gets generated.

NOTE:

Environment used to develop code:

Python 3.6 

Keras library

Windows 10

Imported libraries:

gym, numpy, keras, random, collections, matplotlib.

Code description:

project2.py initiates an agent "DqnAgent" to play the LunarLander game. The agent is trained for 1500 training episodes and tested for next 100 episodes. The plots showing reward per episode and moving average of rewards for both training and tetsing phase get generated.

tune_alpha_vectorised.py initiates 3 agents, "DqnAgent", "DqnAgent1" and "DqnAgent2" to play the LunerLander game with learning rates for their Adam optimizer of Keras Neural Network model as 0.00005, 0.00025 and 0.0005 respectively. The moving average of rewards are plotted for the training phase on the same chart for better comparison.

tune_gamma_vectorised.py initiates 3 agents, "DqnAgent", "DqnAgent1" and "DqnAgent2" to play the LunerLander game with discount factors as 1, 0.99 and 0.79 respectively. The moving average of rewards are plotted for the training phase on the same chart for better comparison.

tune_decay_vectorised.py initiates 3 agents, "DqnAgent", "DqnAgent1" and "DqnAgent2" to play the LunerLander game with epsilon decay rates as  0.999, 0.995 and 0.8 respectively. The moving average of rewards are plotted for the training phase on the same chart for better comparison.

tune_tau_vectorised.py initiates 3 agents, "DqnAgent", "DqnAgent1" and "DqnAgent2" to play the LunerLander game with tau for soft target updates as 0.0001,0.001 and 0.005 respectively. The moving average of rewards are plotted for the training  phase on the same chart for better comparison.

project_2_with_saved_weights.py initiates an agent "DqnAgent" with its neural network model loaded with weights saved in "model_weights2.h5". It runs 100 episodes, where no learning takes place. 

model_weights2.h5 contains the model weights saved from the working model in "project2.py".

project2_DQN_two_networks.py initiates a DqnAgent to play the LunarLander game and uses the best Q value in target network for next state in the Q-value update rule for the current state-action pair, unlike DDQN which uses policy model Q-network for action selection.This was an experiment to improve performance before using DDQN method. DQN_two_networks has two models- input_model and target_model. The target_model gets updated as per soft target update during each time step. 

project2_DQN_one_network.py initiates a DqnAgent to play the LunarLander game and uses only one model for both action selection and Q-value update rule. This vanilla DQN method was attempted before I tried target network and DDQN.