import numpy as np
from skimage.transform import resize

from hakaton.model.transformation import ImageTransformation


class Resize(ImageTransformation):

    def __init__(self, to_width: int, to_height: int):
        self._to_width = to_width
        self._to_height = to_height

    def apply(self, array: np.ndarray) -> np.ndarray:
        return resize(array, output_shape=(self._to_height, self._to_width))
