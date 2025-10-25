import random


class PlaceholderArmEnv:
    def __init__(self, n_arms: int = 5):
        self.n_arms = n_arms
        self.means = [random.random() for _ in range(n_arms)]

    def pull(self, arm: int) -> float:
        return self.means[arm] + random.gauss(0, 0.1)

    def reset(self):
        self.means = [random.random() for _ in range(self.n_arms)]
