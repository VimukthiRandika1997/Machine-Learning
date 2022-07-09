# import gym
# env = gym.make("LunarLander-v2")
# env.action_space.seed(42)

# observation, info = env.reset(seed=42, return_info=True)

# for _ in range(1000):
#     observation, reward, done, info = env.step(env.action_space.sample())
#     env.render()

#     if done:
#         observation, info = env.reset(return_info=True)

# env.close()

import time
import gym
env = gym.make("MountainCar-v0")

# Number of steps you run the agent for
num_steps = 1500

obs = env.reset()

for step in range(num_steps):
    # Take a random action, or an agent can perform more intelligent actions
    action = env.action_space.sample()

    # apply the action
    obs, reward, done, info = env.step(action)

    # render the env
    env.render()

    # wait a bit before the next frame
    time.sleep(0.001)

    # if the episode is up, then start the another one
    if done:
        env.reset()

# close the env
env.close()