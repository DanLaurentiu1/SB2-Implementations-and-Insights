import random


class PlaceholderBanditAgent:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms

    def seed(self, new_seed: int):
        print("Seed was set in the agent!")

    def run_episode(self, env):
        return {"episode_reward": env.pull(1)}
