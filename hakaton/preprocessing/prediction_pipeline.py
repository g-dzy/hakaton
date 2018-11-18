from abc import ABC, abstractmethod
from logging import getLogger
import numpy as np


from sklearn.preprocessing import StandardScaler

from hakaton.model.image import Image
from hakaton.preprocessing.resize import Resize


class PredictionPipeline(ABC):

    def __init__(self,
                 standard_scaler: StandardScaler,
                 resize_to=(320, 256)
                 ):
        self._logger = getLogger(__name__)
        self._standard_scaler = standard_scaler
        self._resize_transformation = Resize(to_width=resize_to[0], to_height=resize_to[1])

    def predict(self, path: str):
        image = Image(path, grayscale=True)
        image.apply_transformation(self._resize_transformation)
        image = self._standard_scaler.transform(image.ndarray)
        return self._predict(image)

    @abstractmethod
    def _predict(self, image: np.ndarray) -> object:
        """

        :param image: a np.ndarray, MxN, M=height, N=width; if gray, this is MxN array, if rgb, this is MxNx3 array
        :return: a prediction
        """

