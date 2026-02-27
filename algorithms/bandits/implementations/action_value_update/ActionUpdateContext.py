from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt


@dataclass
class ActionUpdateContext:
    action: int
    probabilities: Optional[npt.NDArray[np.float64]] = None
