import gym
env = gym.make("Taxi-v3")
env.reset()

state = env.encode(3, 1, 2, 0)
print("State:", state)

env.s = state
env.render()

print(env.P[328]) # Reward Table
# This dictionary has following structure
# {action: [(probability, nextstate, reward, done)]}

print("Action space {}".format(env.action_space))
print("State space {}".format(env.observation_space))

### Implementing Q-Learning ###

import numpy as np

q_table = np.zeros([env.observation_space.n, env.action_space.n])

"""Training the agent"""

import random

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# For plotting metrics
all_epochs = []
all_penalties = []

# Establishing Q-table over 100,000 episodes.
for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values
        
        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

print("Training finished.\n")
print(q_table[328])


"""Evaluate agent's performance after Q-learning"""

total_episodes, total_penalties, total_epochs = 0, 0, 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False

    while not done:
        action = np.argmax(q_table[state])
        state, reward, done, info = env.step(action)

        if reward == -10:
            penalties += 1
        
        epochs += 1

        total_penalties += penalties
        total_epochs += epochs

print(f"Results after {episodes} episodes")
print(f"Average timesteps per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")