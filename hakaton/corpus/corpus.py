import numpy as np

import pandas as pd

import os

from typing import List, Tuple

from sklearn.utils import shuffle

from hakaton.corpus.columns import Columns
from hakaton.img.img_reader import ImgReader


class Corpus:
    def __init__(self, img_reader: ImgReader, metadata_file_path: str, root_dir: str):
        self._img_reader = img_reader
        self._metadata_file_path = metadata_file_path
        self._metadata_df = self.__read_metadata()
        self._root_dir = root_dir

    def get_all_images_with_column(self, col_name: str, is_shuffle=False):
        df = self._metadata_df[[Columns.FILENAME.value, col_name]]
        if is_shuffle:
            df = shuffle(df)
        files = self.__list_files_from_df(df)
        images = self._img_reader.read_multiple(files)
        col_values = df[col_name].values

        return images, col_values

    def get_all_type(self, is_shuffle=False) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_all_images_with_column(Columns.TYPE.value, is_shuffle)

    def get_all_uic(self, is_shuffle=False) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_all_images_with_column(Columns.UIC_VALUE.value, is_shuffle)

    def get_by_type(self, type_: int) -> np.ndarray:
        """"
        Get images by type. Type meaning:
        0 - wagon
        1 - space between wagon
        """
        df = self._metadata_df[self._metadata_df[Columns.TYPE.value] == type_]
        files = self.__list_files_from_df(df)
        images = self._img_reader.read_multiple(files)

        return images

    def get_by_uic(self, uic: int):
        """
        Get images with (uic == 1) or without uic (uic == 0) code.
        :param uic:
        :return:
        """
        df = self._metadata_df[self._metadata_df[Columns.IS_UIC.value] == uic]
        files = self.__list_files_from_df(df)
        images = self._img_reader.read_multiple(files)

        return images

    def __list_dir_files(self) -> List[str]:
        files = os.listdir(self._root_dir)

        return files

    def __list_files_from_df(self, df: pd.DataFrame):
        files = [os.path.join(self._root_dir, f) for f in df[Columns.FILENAME.value].values]

        return files

    def __read_metadata(self) -> pd.DataFrame:
        df = pd.read_csv(self._metadata_file_path)

        return df
