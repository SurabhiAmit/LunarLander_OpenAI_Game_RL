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
        self.test_mode = False
        self.train_mode = True
        self.total_reward = 0
        self.test_mode_rewards = {}
        self.train_mode_rewards = {}
        self.tau = tau
        self.use_soft_update = False
        self.max_storable= 100000
        self.min_storable = 10000


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

    def experience_replay(self, num_episode):
        if self.min_storable > len(self.storage):
            return
        samples = np.array(random.sample(self.storage, self.mini_batch_size))
        # Code reference for function_approximation code : GitHub. (2018). mimoralea/applied-reinforcement-learning. [online]
        # Available at: https://github.com/mimoralea/applied-reinforcement-learning/tree/master /notebooks/solutions.
        state_batch = np.array(samples[:, 0].tolist())
        action_batch = np.array(samples[:, 1].tolist())
        reward_batch = np.array(samples[:, 2].tolist())
        next_state_batch = np.array(samples[:, 3].tolist())
        done_samples = samples[np.where(samples[:, 4] == True)]
        not_done_samples = samples[np.where(samples[:, 4] == False)]
        improved_estimate = np.empty(samples.shape[0])
        Q_state_batch = np.array(self.input_model.predict(state_batch))
        if not_done_samples.size != 0:
            #best_next_action_ddqn_batch = np.array(np.argmax(np.array(self.input_model.predict(next_state_batch[np.where(samples[:, 4] == False)])),axis=1))
            #target_array = np.array(self.target_model.predict(next_state_batch[np.where(samples[:, 4] == False)]))
            improved_estimate[np.where(samples[:, 4] == False)] = not_done_samples[:, 2] + self.gamma * \
                                                                                           np.array(np.max(self.input_model.predict(next_state_batch[np.where(samples[:, 4] == False)]),axis=1))
        improved_estimate[np.where(samples[:, 4] == True)] = done_samples[:, 2]
        improved_estimate.reshape((-1, 1))
        Q_state_batch[np.arange(Q_state_batch.shape[0]),action_batch] = improved_estimate
        self.input_model.fit(state_batch, Q_state_batch, batch_size=self.mini_batch_size, epochs=1, verbose=False)

#plotting functions

def plot_train(reward_dict):
    if not reward_dict:
        return
    sorted_list = sorted(reward_dict.items())
    episode, reward = zip(*sorted_list)
    plt.plot(episode, reward, label="Reward in each episode")
    window = 100
    sma = {}
    key_list = list(reward_dict.keys())
    key_list.sort()
    first = key_list[0]
    last = key_list[-1]
    def average(k):
        sum = 0
        for j in range(k - window + 1, k + 1):
            sum += reward_dict[j]
        return sum / window
    for i in range(first + window - 1, last + 1):
        sma[i] = average(i)
    sorted_list1 = sorted(sma.items())
    if sorted_list1:
        episode1, reward1 = zip(*sorted_list1)
        plt.plot(episode1, reward1, label="Moving average with a window of 100 episodes")
    plt.title("Rewards during training episodes of LunarLander")
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()

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

def plot_reward(reward_dict, title):
    if not reward_dict:
        return
    sorted_list = sorted(reward_dict.items())
    episode, reward = zip(*sorted_list)
    plt.plot(episode, reward, label="Reward in each episode")
    sum = 0
    for each in list(reward_dict.keys()):
        sum += reward_dict[each]
    average = sum / len(list(reward_dict.keys()))
    plt.title(title)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.axhline(y=average, color='r', label='Average Reward')
    plt.show()

#initialization of the agent and the environment

episodes = 1600
limit = 1500
env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n
DqnAgent = Dqn(state_dim, num_actions, alpha=0.0005, epsilon_decay=0.995, gamma=0.99, tau=0.001)
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
    if i > limit:
        DqnAgent.test_mode = True
        DqnAgent.train_mode = False
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
        if DqnAgent.train_mode:
            DqnAgent.add_exp(current_state, current_action, reward, next_state, done)
            DqnAgent.experience_replay(i)
        current_state = next_state
    if DqnAgent.test_mode:
        DqnAgent.test_mode_rewards[k] = DqnAgent.total_reward
        k += 1
    if DqnAgent.train_mode:
        DqnAgent.train_mode_rewards[j] = DqnAgent.total_reward
        j += 1
    print("Episode is : ", i, "DONE at step count of ", step_count, "last reward is : ", reward,
          " train mode is : ", DqnAgent.train_mode, " Total rewards is:", DqnAgent.total_reward, " epsilon is : ",
          DqnAgent.epsilon)
    #additional plotting for each 100 episodes in training set..extra debugging
    if i < 100:
        train1[i] = DqnAgent.total_reward
    elif i < 200:
        train2[i] = DqnAgent.total_reward
    elif i < 300:
        train3[i] = DqnAgent.total_reward
    elif i < 400:
        train4[i] = DqnAgent.total_reward
    elif i < 500:
        train5[i] = DqnAgent.total_reward
    elif i < 600:
        train6[i] = DqnAgent.total_reward
    elif i < 700:
        train7[i] = DqnAgent.total_reward
    elif i < 800:
        train8[i] = DqnAgent.total_reward
    elif i < 900:
        train9[i] = DqnAgent.total_reward
    elif i < 1000:
        train10[i] = DqnAgent.total_reward

#saving weights
DqnAgent.input_model.save_weights('model_weights.h5')

#Plotting graphs included in the report
plot_train(DqnAgent.train_mode_rewards)
plot_test(DqnAgent.test_mode_rewards)

#extra debugging..please uncomment this to see the plots of rewards in batches of 100 training episodes for the 1000 training episodes
'''plot_reward(train1, title="Training mode of LunarLander episode 1 to 100")
plot_reward(train2, title="Training mode of LunarLander episode 100 to 200")
plot_reward(train3, title="Training mode of LunarLander episode 200 to 300")
plot_reward(train4, title="Training mode of LunarLander episode 300 to 400")
plot_reward(train5, title="Training mode of LunarLander episode 400 to 500")
plot_reward(train6, title="Training mode of LunarLander episode 500 to 600")
plot_reward(train7, title="Training mode of LunarLander episode 600 to 700")
plot_reward(train8, title="Training mode of LunarLander episode 700 to 800")
plot_reward(train9, title="Training mode of LunarLander episode 800 to 900")
plot_reward(train10, title="Training mode of LunarLander episode 900 to 1000")'''