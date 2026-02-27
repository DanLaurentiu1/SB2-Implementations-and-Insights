import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def get_last_average(input_folder: Path, keyword: str, output_csv: Path):
    memory, counter = np.float64(0), 0
    all_csvs: List[Path] = []

    for file_path in input_folder.iterdir():
        if keyword in file_path.name:
            all_csvs.append(file_path)

    print(f"{len(all_csvs)} csvs were found!")
    if not all_csvs:
        raise ValueError(
            f"No CSV files found in {input_folder} containing keyword={keyword}"
        )

    for file_path in all_csvs:
        counter += 1
        df = pd.read_csv(file_path)
        last_row = df.to_numpy()[-1]

        last_average_reward = last_row[4]
        memory = memory + ((last_average_reward - memory) / counter)

    df_out = pd.DataFrame({"avg_reward": [memory]})
    df_out.to_csv(output_csv)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m path.to.file <epsilon>")

    get_last_average(
        input_folder=Path(
            "algorithms/bandits/experiments/Figure_2.6_Green___24February2026/raw"
        ),
        keyword=sys.argv[1],
        output_csv=Path(
            f"algorithms/bandits/experiments/Figure_2.6_Green___24February2026/processed/aggregated_over_{sys.argv[1]}.csv"
        ),
    )
