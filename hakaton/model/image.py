from skimage.io import imread, imsave
import os
from hakaton.model.transformation import ImageTransformation


class Image:

    def __init__(self, file_path: str, grayscale: bool=False):
        self._data = imread(file_path, as_gray=grayscale)

    def apply_transformation(self, transformation: ImageTransformation):
        self._data = transformation.apply(self._data)

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
        imsave(path, self._data)
