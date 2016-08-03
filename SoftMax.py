import math
import random

from BernoulliArm import BernoulliArm
from simulation import test_alogorithm

def categorical_draw (probs):
    z = random.random()
    cum_prob = 0.0
    for i in range(len(probs)):
        prob = probs[i]
        cum_prob += prob
        if cum_prob > z:
            return i
    return len(probs) - 1

class SoftMax:
    def __init__ (self, temperature, counts, values):
        self.temperature = temperature
        self.counts = counts
        self.values = values
        return

    def initialize (self, n_arms):
        self.counts = [0 for col in range(n_arms)]
        self.values = [0.0 for col in range(n_arms)]
        return

    def select_arm (self):
        z = sum([math.exp(v / self.temperature) for v in self.values])
        probs = [math.exp(v / self.temperature) for v in self.values]
        return categorical_draw(probs)

    def update (self, chosen_arm, reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
        n = self.counts[chosen_arm]

        value = self.values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm] = new_value
        return


def ind_max (x):
    m = max(x)
    return x.index(m)


random.seed(1)
means = [0.1, 0.1, 0.1, 0.1, 0.9]
n_arms = len(means)
random.shuffle(means)

def cb (mu):
    return BernoulliArm(mu)

arms = list(map(cb, means))

print("Best arm is " + str(ind_max(means)))
f = open("standard_softmax.csv", "w")

for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
    algo = SoftMax(epsilon, [], [])
    algo.initialize(n_arms)
    results = test_alogorithm(algo, arms, 5000, 250)
    for i in range(len(results[0])):
            f.write(str(epsilon) + ",\t ")
            f.write(",\t".join([str(results[j][i]) for j in  range(len(results))]) + "\n")
f.close()


