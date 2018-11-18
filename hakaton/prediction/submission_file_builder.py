from enum import Enum
from logging import getLogger

import pandas as pd
import os


class SubmissionFileColumns(Enum):
    TEAM_NAME = "team_name"
    TRAIN_NUMBER = "train_number"
    LEFT_RIGHT = "left_right"
    FRAME_NUMBER = "frame_number"
    WAGON = "wagon"
    UIC_0_1 = "uic_0_1"
    UIC_LABEL = "uic_label"


class SubmissionFileBuilder:

    def __init__(self, team_name: str):

        self._logger = getLogger()
        assert self._team_name

        columns = [
            SubmissionFileColumns.TEAM_NAME,
            SubmissionFileColumns.TRAIN_NUMBER,
            SubmissionFileColumns.LEFT_RIGHT,
            SubmissionFileColumns.FRAME_NUMBER,
            SubmissionFileColumns.WAGON,
            SubmissionFileColumns.UIC_0_1,
            SubmissionFileColumns.UIC_LABEL
        ]

        self._df = pd.DataFrame(columns=columns)
        self._team_name = team_name

    def append_prediction(self, train_number: str, left_or_right: str, 
                          frame_number: int, wagon: str, uic_01: int, uic_label: str):

        assert left_or_right in ['left', 'right']
        assert frame_number >= 0
        assert uic_01 in [0, 1]
        if uic_label:
            # TODO warn if control sum is invalid
            # TODO throw if pattern is invalid
            assert uic_label

        row = {
            SubmissionFileColumns.TEAM_NAME: self._team_name,
            SubmissionFileColumns.TRAIN_NUMBER: train_number,
            SubmissionFileColumns.LEFT_RIGHT: left_or_right,
            SubmissionFileColumns.FRAME_NUMBER: frame_number,
            SubmissionFileColumns.WAGON: wagon,
            SubmissionFileColumns.UIC_0_1: uic_01,
            SubmissionFileColumns.UIC_LABEL: uic_label
        }

        self._df = self._df.append(row, ignore_index=True)

    def to_csv(self, path: str):
        if not os.path.exists(os.path.dirname(path)):
            self._logger.info('Saving csv to %s' % path)
            os.makedirs(os.path.dirname(path))
        self._df.to_csv(path, index=False)
