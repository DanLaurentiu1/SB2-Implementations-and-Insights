from pathlib import Path
import csv
from typing import List


class CSVLogger:
    def __init__(
        self, directory: Path, columns: List[str], filename: str = "results.csv"
    ):
        if not columns:
            raise ValueError("`columns` must be a non-empty list of column names.")

        self.directory = directory
        self.columns = columns
        self.filepath = self.directory / filename
        self._write_header()

    def _write_header(self):
        with self.filepath.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()

    def log(self, row: dict[str, object]):
        with self.filepath.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writerow(row)

    def get_path(self) -> Path:
        return self.filepath
