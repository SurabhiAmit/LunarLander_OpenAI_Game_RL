import random
import gym
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import matplotlib.pyplot as plt

# code reference for Source code of LunarLander environment: GitHub. (2018). openai/gym. [online] Available at: https://github.com/openai/gym.

class Dqn:
    def __init__(self,states_count,actions_count,alpha,epsilon_decay,gamma,tau):
        self.states_count = states_count
        self.actions_count = actions_count
        self.alpha = alpha
        self.decay_rate_alpha = 0.0
        self.epsilon = 1.0
        self.decay_epsilon_rate = epsilon_decay
        self.gamma = gamma
        self.storage = deque()
        self.mini_batch_size = 32
        self.min_epsilon = 0.01
        self.input_model = self.create_model()
        self.test_mode = True
        self.train_mode = False
        self.total_reward = 0
        self.test_mode_rewards = {}

    def create_model(self):
        #Neural network + DQN
        model = Sequential()
        model.add(Flatten(input_shape=(1,self.states_count)))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(self.actions_count))
        model.add(Activation('linear'))
        optimize = Adam(lr=self.alpha, decay=0.0)
        model.compile(optimizer=optimize, loss='mean_squared_error')
        #model.summary()
        return model

    def add_exp(self,state,action,reward,next_state,done):
        self.storage.append((state,action,reward,next_state,done))
        if len(self.storage) > self.max_storable:
            self.storage.popleft()

    def take_action(self,state):
        if self.train_mode:
            if random.random() <= self.epsilon:
                return random.randint(0,self.actions_count-1)
        state = np.array([state])
        return np.argmax(self.input_model.predict(state)[0])

#plotting functions

def plot_test(dict):
    sorted_list = sorted(dict.items())
    episode, reward = zip(*sorted_list)
    plt.plot(episode, reward, label='Reward')
    sum = 0
    for each in list(dict.keys()):
        sum += dict[each]
    average = sum / 100
    plt.title("Rewards during test episodes of Lunarlander")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.axhline(y=average, color='r', label='Average Reward for 100 test episodes')
    plt.legend()
    plt.show()

#initialization of the agent and the environment

episodes = 100
env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
DqnAgent = Dqn(state_dim, num_actions, alpha=0.0005, epsilon_decay=0.995, gamma=0.99, tau=0.001)
DqnAgent.input_model.load_weights('model_weights2.h5')
done = False
j, k = 0, 0
train1, train2, train3, train4, train5, train6, train7, train8, train9, train10 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
seed = 1
np.random.seed(seed)
random.seed(seed)
env.seed(seed)
print("Vectorised project2.py with alpha=0.0005, epsilon_decay=0.995, gamma=0.99, tau=0.001")
for i in range(episodes):
    DqnAgent.total_reward = 0
    DqnAgent.epsilon = DqnAgent.epsilon * DqnAgent.decay_epsilon_rate
    current_state = env.reset()
    target_update_count = 50
    step_count = 0
    done = False
    while not done:
        current_state = np.reshape(current_state, [1, state_dim])
        step_count += 1
        current_action = DqnAgent.take_action(current_state)
        next_state, reward, done, info = env.step(current_action)
        DqnAgent.total_reward += reward
        next_state = np.reshape(next_state, [1, state_dim])
        current_state = next_state
    if DqnAgent.test_mode:
        DqnAgent.test_mode_rewards[k] = DqnAgent.total_reward
        k += 1
    print("Episode is : ", i, "DONE at step count of ", step_count, "last reward is : ", reward,
          " train mode is : ", DqnAgent.train_mode, " Total rewards is:", DqnAgent.total_reward, " epsilon is : ",
          DqnAgent.epsilon)

#Plotting graphs included in the report
plot_test(DqnAgent.test_mode_rewards)