import numpy as np

import pandas as pd

import os

from typing import List

from hakaton.corpus.columns import Columns
from hakaton.img.img_reader import ImgReader


class Corpus:
    def __init__(self, img_reader: ImgReader, metadata_file_path: str, root_dir: str):
        self._img_reader = img_reader
        self._metadata_file_path = metadata_file_path
        self._metadata_df = self.__read_metadata()
        self._root_dir = root_dir

    def get_by_type(self, type_: int) -> np.ndarray:
        """"
        Get images by type. Type meaning:
        0 - wagon
        1 - space between wagon
        """
        df = self._metadata_df[self._metadata_df[Columns.TYPE.value] == type_]
        files = [os.path.join(self._root_dir, f) for f in df[Columns.FILENAME.value].values]
        images = self._img_reader.read_multiple(files)

        return images

    def get_by_uic(self, uic: int):
        """
        Get images with (uic == 1) or without uic (uic == 0) code.
        :param uic:
        :return:
        """
        df = self._metadata_df[self._metadata_df[Columns.IS_UIC.value] == uic]
        files = [os.path.join(self._root_dir, f) for f in df[Columns.FILENAME.value].values]
        images = self._img_reader.read_multiple(files)

        return images

    def __list_dir_files(self) -> List[str]:
        files = os.listdir(self._root_dir)

        return files

    def __read_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self._metadata_file_path)

        return df
