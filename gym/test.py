import gym
import random
import matplotlib.pyplot as plt

"""Create our environment"""
env = gym.make("BreakoutDeterministic-v4", render_mode="human")

"""Reset the environment, returns to the first frame of the game"""
first_frame = env.reset()
plt.imshow(first_frame)

"""Now we can take actions using env.step function. In breakout the possible actions are

    0 = Stay Still
    1 = Start Game/Shoot Ball
    2 = Move Right
    3 = Move Left

"""

"""Receiving next_frame, next_frame_reward, done, info"""
next_frame, next_frame_reward, done, info = env.step(1)
plt.imshow(next_frame)
print("Reward Received: ", str(next_frame_reward))
print("Next State is a terminal state: ", str(done))
print("Info: ", str(info))


"""Now lets take a bounch of random actions and watch the gameplay using render() method"""
for i in range(10000):
    action = random.sample([0, 1, 2, 3], 1)[0]
    observation, reward, done, info = env.step(action)
    env.render(mode="rgb_array")

env.close()


import cv2
import numpy as np


def resize_frame(frame):
    frame = frame[30:-12,5:-4]
    frame = np.average(frame,axis = 2)
    frame = cv2.resize(frame,(84,84),interpolation = cv2.INTER_NEAREST)
    frame = np.array(frame,dtype = np.uint8)
    return frame