from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_optimal_chosen_percentage(folder_path: Path):
    data_map = {
        "aggregated_over_NormalAVInit.csv": (
            "#919498",
            600,
            45,
            r"Realistic, $\epsilon$-greedy\n$Q_1=0, \epsilon=0.1$",
        ),
        "aggregated_over_OptimisticAVInit.csv": (
            "#0097d7",
            750,
            70,
            r"Optimistic, greedy\n$Q_1=5, \epsilon=0.1$",
        ),
        # "aggregated_over_0.9.csv": ("green", 9000, 0.385, "$\\alpha=0.9$"),
        # "aggregated_over_0.02.csv": ("orange", 7000, 0.5, "$\\alpha=0.02$"),
    }
    all_csvs: List[Path] = []
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        }
    )
    plt.figure(figsize=(8, 5))

    for csv_file in folder_path.iterdir():
        all_csvs.append(csv_file)

    for csv_file in all_csvs:
        rows = pd.read_csv(csv_file).to_numpy()
        plt.plot(rows[:, 0], rows[:, 2] * 100, color=data_map[csv_file.name][0])
        plt.text(
            data_map[csv_file.name][1],
            data_map[csv_file.name][2],
            data_map[csv_file.name][3],
            color=data_map[csv_file.name][0],
            fontsize=12,
        )

    plt.yticks(np.arange(0, 100 + 1, 20))
    plt.xticks(np.arange(0, 1000 + 1, 200))
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.savefig(folder_path / "new_plot.pdf", format="pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    plot_optimal_chosen_percentage(
        folder_path=Path(
            "algorithms/bandits/experiments/Figure_2.3___21December2025/processed"
        )
    )
