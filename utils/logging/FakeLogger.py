from typing import Dict, Any
from utils.logging.BaseLogger import BaseLogger


class FakeLogger(BaseLogger):
    def __init__(self):
        self.rows = []

    def log(self, row: Dict[str, Any]):
        self.rows.append(dict(row))

    def get_rows(self):
        return self.rows

    def get_path(self):
        return None
