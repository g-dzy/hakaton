from typing import List

import numpy as np

from hakaton.prediction.model import SkyhacksModel
from hakaton.util import model_util


class WagonDetectorSkyhacksModel(SkyhacksModel):
    MODEL_STRUCTURE_FILE = "storedmodel/model-next-wagon-structure.json"
    MODEL_WEIGHTS_FILE = "storedmodel/model-next-wagon-weights.h5"

    def __init__(self, frame_cnt_required=3):
        self._model = model_util.load(self.MODEL_STRUCTURE_FILE, self.MODEL_WEIGHTS_FILE)
        self._frame_cnt_required = frame_cnt_required

    def predict(self, train_images: List[np.ndarray], batch_size=None) -> List[object]:
        x = np.asarray(train_images)
        x = x.reshape(x.shape[0], -1)
        predicted = self._model.predict(x, batch_size)
        labels = self._parse_next_wagon_prediction(predicted, self._frame_cnt_required)

        return labels.tolist()

    def _parse_next_wagon_prediction(self, predicted: np.ndarray, frame_cnt_required=2):
        wagon_numbers = list()
        current_wagon_num = 0
        frame_cnt = 0
        found_locomotive = False
        for i, label in enumerate(predicted):
            if (label == 1):
                frame_cnt += 1
            else:
                frame_cnt = 0

            if (frame_cnt == frame_cnt_required):
                if found_locomotive:
                    current_wagon_num += 1
                    wagon_numbers[-frame_cnt_required + 1:] = [current_wagon_num for i in range(frame_cnt_required - 1)]
                else:
                    found_locomotive = True
            wagon_numbers.append(current_wagon_num)

        return np.array(wagon_numbers)
