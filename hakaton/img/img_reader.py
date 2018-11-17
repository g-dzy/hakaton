from typing import List

import numpy as np

import matplotlib.pylab as plt


class ImgReader:
    def read(self, file: str, format_="jpg") -> np.ndarray:
        return plt.imread(file, format_)

    def read_multiple(self, files: List[str], format_="jpg") -> np.ndarray:
        return np.asarray([self.read(f, format_) for f in files])
