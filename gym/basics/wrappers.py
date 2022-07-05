import gym
from gym.wrappers import RescaleAction

base_env = gym.make("BipedalWalker-v3")
print(base_env.action_space)

wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
print(wrapped_env.action_space)