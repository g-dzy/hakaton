import os
from hakaton.model.transformation import ImageTransformation
from matplotlib import pyplot as plt
import scipy.ndimage


class Image:

    def __init__(self, file_path: str, grayscale: bool=False):
        self._data = scipy.ndimage.imread(file_path, mode='L' if grayscale else None, flatten=grayscale)

    def apply_transformation(self, transformation: ImageTransformation):
        self._data = transformation.apply(self._data)
        return self

    def apply_lambda_transformation(self, transformation_func):
        self._data = transformation_func(self._data)
        return self

    @property
    def ndarray(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def size(self):
        return tuple(self._data.shape[:2])

    @property
    def width(self):
        return self._data.shape[1]

    @property
    def height(self):
        return self._data.shape[0]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.imsave(path, self._data)
