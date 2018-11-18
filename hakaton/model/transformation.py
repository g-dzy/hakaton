from abc import ABC, abstractmethod
import numpy as np


class ImageTransformation(ABC):

    @abstractmethod
    def apply(self, array: np.ndarray) -> np.ndarray:
        pass
