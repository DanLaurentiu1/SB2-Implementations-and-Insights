from typing import List


class PlaceholderBanditAgent:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self._seed = 0
        self._metrics_to_track = ["episode", "episode_reward"]

    def set_seed(self, new_seed: int):
        self._seed = new_seed

    def run_episode(self, env):
        return {"episode_reward": env.pull(1)}

    def get_metrics(self) -> List[str]:
        return self._metrics_to_track
