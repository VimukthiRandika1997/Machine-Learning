import gym
env = gym.make("MountainCar-v0")

# Observation and action spaces
obs_space = env.observation_space
action_space = env.action_space
print("The observation space: {}".format(obs_space))
print("The action space: {}".format(action_space))

import matplotlib.pyplot as plt

# reset the environment and see the initial observation
obs = env.reset()
print("The initial observation is {}".format(obs))

# Sample a random action from the entire action space
random_action = env.action_space.sample()

# Take the action and get the new observation space
new_obs, reward, done, info = env.step(random_action)
print("The new observation is {}".format(new_obs))

# seeing the current state of the environment
env.render(mode="human")
# env.close() # for closing the window

# Seeing the screenshot of the game
env_screen = env.render(mode='rgb_array')
env.close()

plt.imshow(env_screen)

print("Upper Bound for Env Observation", env.observation_space.high)
print("Lower Bound for Env Observation", env.observation_space.low)