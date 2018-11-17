import logging
from math import ceil
from typing import List
import os
import glob
from sklearn.preprocessing import StandardScaler
from functools import reduce
from sklearn.externals import joblib
import numpy as np

from hakaton.model.image import Image


class Standardizer:

    def __init__(self):

        self._scaler = StandardScaler()
        self._logger = logging.getLogger(__name__)

    def fit_transform(self, output_directory: str, base_directories: List[str]):

        self._logger.info('Listing directories: %s' % '\n'.join(base_directories))

        files = list(
            reduce(lambda a, b: list(a) + list(b), [
                glob.iglob(os.path.join(directory, "**", "*.jpg"), recursive=True) for directory in base_directories
            ])
        )

        files_count = len(files)

        for i, file in enumerate(files):
            if i % 50 == 0:
                self._logger.info('File %d of %d' % (i+1, files_count))
            self._scaler.partial_fit(Image(file, grayscale=True).ndarray)
            i += 1

        for i, file in enumerate(files):
            if i % 50 == 0:
                self._logger.info('File %d of %d' % (i+1, files_count))
            output_path = os.path.join(output_directory, file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image = Image(file, grayscale=True)
            image.apply_lambda_transformation(self._scaler.transform)
            image.save(output_path)

        return self._scaler

    def save(self, path: str):

        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._scaler, path)

    @classmethod
    def load(cls, path: str):

        return joblib.load(path)
