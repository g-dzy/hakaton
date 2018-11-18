from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SkyhacksModel(ABC):

    @abstractmethod
    def predict(self, train_images: List[np.ndarray]) -> List[object]:
        """

        :param train_images: 320x256 0-255 grayscale images, ordered by frame number
        :return:
        """
        raise NotImplemented()
