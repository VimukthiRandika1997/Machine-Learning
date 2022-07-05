import gym
import pygame
from gym.utils.play import play

# plotting real time statistics
def callback(obs_t, obs_tp1, action, rew, done, info):
    return [rew,]


mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}
play(gym.make("CartPole-v1"), keys_to_action=mapping)