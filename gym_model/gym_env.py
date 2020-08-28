import gym
import matplotlib.pyplot as plt
import numpy as np
from gym_model.Agent import GymAgent


def run_model():
    agent = GymAgent(ALPHA=0.0005, GAMMA=0.99, n_actions=4,
                  layer1_size=64, layer2_size=64, input_dims=8, fname='model.h5')
    env = gym.make('LunarLander-v2')
    score_history = []

    n_episodes = 2000

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        agent.learn()

        print('episode', i, 'score %.1f' % score, 'average_score %.1f' %
              np.mean(score_history[-100:]))
    agent.save_model()


if __name__ == '__main__':
    run_model()
