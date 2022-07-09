import gym
env = gym.make("ALE/Alien-v5")
observation, info = env.reset(seed=42, return_info=True)


for _ in range(1000):
    env.render()
    #action = policy(observation)  # User-defined policy function
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)

    if done:
        observation, info = env.reset(return_info=True)
env.close()