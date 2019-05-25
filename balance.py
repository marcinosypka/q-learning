import math

import gym
import numpy as np


class QLearner:
    def __init__(self, buckets=(1, 1, 6, 12,), alpha=0.9, epsilon=0.1, gamma=1.0):
        self.environment = gym.make('CartPole-v1')
        self.attempt_no = 1
        self.buckets = buckets
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            0.5,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            -0.5,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.Q = np.zeros(self.buckets + (self.environment.action_space.n,))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.scores = []

    def learn(self, max_attempts):
        reward_sum = 0
        for t in range(max_attempts):
            reward_sum = self.attempt(t)
            #print(reward_sum)
            # if reward_sum >= 500:
            #     # self.save_results(
            #     #     "results2/results_attemps_{}_a{}_e{}_g{}_b{}{}{}{}.csv".format(t,self.alpha, self.epsilon, self.gamma, self.buckets[0],
            #     #                                                self.buckets[1], self.buckets[2], self.buckets[3]))
            #     self.save_results(
            #         "results_attemps_{}.csv".format(t))
            #     break
            self.save_results(
                    "results_attemps_10.csv")
        print("SCORE: {}, alpha: {} gamma: {}, epsilon: {}, buckets: [{} {} {} {}]".format(reward_sum, self.alpha,self.gamma,self.epsilon,self.buckets[0],self.buckets[1],self.buckets[2],self.buckets[3]))
        return max

    def attempt(self,t):
        observation = self.discretise(self.environment.reset())
        done = False
        reward_sum = 0.0
        while not done:
            # self.environment.render()
            action = self.pick_action(observation)
            new_observation, reward, done, info = self.environment.step(action)
            new_observation = self.discretise(new_observation)
            self.update_knowledge(action, observation, new_observation, reward)
            observation = new_observation
            reward_sum += reward
        self.scores.append(reward_sum)
        self.attempt_no += 1
        return reward_sum

    def save_results(self,filename):
        import csv
        with open(filename, 'w') as csvfile:
            fieldnames = ['No', 'score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i,score in enumerate(self.scores):
                writer.writerow({"No":i,"score": score})

    def discretise(self, observation):
        ratios = [(observation[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i]) for i in range(len(observation))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(observation))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(observation))]
        return tuple(new_obs)

    def pick_action(self, observation):
        return self.environment.action_space.sample() if (np.random.random() <= self.epsilon) else np.argmax(self.Q[observation])

    def update_knowledge(self, action, observation, new_observation, reward):
        self.Q[observation][action] += self.alpha* (
                reward + self.gamma * np.max(self.Q[new_observation]) - self.Q[observation][action])


def main():
    np.random.seed(0)
    # parameters = [[0.1,0.2,0.4,0.6,0.8],[0.1,0.2,0.4,0.6,0.8],[0.1,0.5,0.9,1],[(1,1,6,6),(8,8,8,8),(2,2,10,10),(4,4,10,10),(1,1,10,10)]]
    #
    # for a in parameters[0]:
    #     for e in parameters[1]:
    #         for g in parameters[2]:
    #             for b in parameters[3]:
    #                 learner = QLearner(b,a,e,g)
    #                 learner.learn(1000)

    learner = QLearner((1,1,6,6), 0.1, 0.4, 1)
    learner.learn(1000)
if __name__ == '__main__':
    main()
