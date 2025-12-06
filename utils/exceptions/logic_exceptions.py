class LogicException(Exception):
    def __init__(self, message: str):
        super().__init__(f"[LogicException] {message}")


class EnvironmentLogicException(LogicException):
    def __init__(self, message: str):
        super().__init__(f"[EnvironmentLogicException] {message}")


class DriftingLogicException(LogicException):
    def __init__(self, message: str):
        super().__init__(f"[DriftingLogicException] {message}")


class RewardLogicException(LogicException):
    def __init__(self, message: str):
        super().__init__(f"[RewardLogicException] {message}")


class ExplorationLogicException(LogicException):
    def __init__(self, message: str):
        super().__init__(f"[ExplorationLogicException] {message}")


class ActionValueLogicException(LogicException):
    def __init__(self, message: str):
        super().__init__(f"[ActionValueLogicException] {message}")
