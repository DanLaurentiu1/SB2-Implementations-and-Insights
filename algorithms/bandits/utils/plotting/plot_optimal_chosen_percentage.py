from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_optimal_chosen_percentage(folder_path: Path):
    data_map = {
        "aggregated_over_AverageSampling.csv": ("red", 9000, 1.5, "$\\alpha=1/n$"),
        "aggregated_over_ERW.csv": ("blue", 6000, 1.7, "$\\alpha=1/10$"),
        # "aggregated_over_eps=0.csv": ("green", 850, 0.925, 0),
    }
    all_csvs: List[Path] = []

    plt.figure(figsize=(8, 5))

    for csv_file in folder_path.iterdir():
        all_csvs.append(csv_file)

    for csv_file in all_csvs:
        rows = pd.read_csv(csv_file).to_numpy()
        plt.plot(rows[:, 0], rows[:, 2], color=data_map[csv_file.name][0])
        plt.text(
            data_map[csv_file.name][1],
            data_map[csv_file.name][2],
            data_map[csv_file.name][3],
            color=data_map[csv_file.name][0],
            fontsize=12,
        )

    plt.yticks(np.arange(0, 1 + 0.2, 0.2))
    plt.xticks(np.arange(0, 1000 + 1, 250))
    plt.xlabel("Steps")
    plt.ylabel("Opitmal Chosen")
    plt.show()


if __name__ == "__main__":
    plot_optimal_chosen_percentage(
        folder_path=Path(
            "algorithms/bandits/experiments/testing_something_08_December_2025/processed"
        )
    )
