from skimage.io import imread


class Image:

    def __init__(self, file_path: str, grayscale: bool=False):
        self._data = imread(file_path, as_gray=grayscale)

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return tuple(self._data.shape[:2])

    @property
    def width(self):
        return self._data.shape[1]

    @property
    def height(self):
        return self._data.shape[0]
