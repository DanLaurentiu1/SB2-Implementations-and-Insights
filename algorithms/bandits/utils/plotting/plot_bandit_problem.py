from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from hydra import main
from matplotlib.pylab import norm
from omegaconf import DictConfig

from environments.make_env import make_env

CONFIGS_DIR = Path(__file__).parents[2] / "configs"


@main(
    config_path=str(CONFIGS_DIR),
    config_name="default",
    version_base=None,
)
def plot_bandit_problem(cfg: DictConfig):
    env_seed = int(cfg["run"]["env_seed"])
    env = make_env(cfg["environment"], seed=env_seed)
    arm_means = env.arm_means
    std = 1.0
    y = np.linspace(-4, 4, 400)
    plt.figure(figsize=(12, 6))
    for action, mean in enumerate(arm_means, start=1):
        pdf = norm.pdf(y, loc=mean, scale=std) * 0.4
        plt.fill_betweenx(y, action - pdf, action + pdf, color="#969696")
        plt.plot(
            [action - 0.2, action + 0.2], [mean, mean], color="black", linewidth=0.9
        )
        plt.text(
            action + 0.25,
            mean,
            r"$q_{*}(%d)$" % action,
            fontsize=10,
            verticalalignment="center",
            horizontalalignment="left",
            color="black",
        )

    plt.axhline(0, color="black", linewidth=0.9, linestyle=(0, (8, 7)))
    plt.xlim(0.5, len(arm_means) + 0.5)
    plt.xticks(range(1, len(arm_means) + 1))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.show()


if __name__ == "__main__":
    plot_bandit_problem()
