import numpy as np

from typing import List

from hakaton.prediction.model import SkyhacksModel
from hakaton.util import model_util


class UicDetectorSkyhacksModel(SkyhacksModel):
    MODEL_STRUCTURE_FILE = "model-uic-detector-structure.json"
    MODEL_WEIGHTS_FILE = "model-uic-detector-weights.h5"

    def __init__(self, frame_cnt_required=3):
        self._model = model_util.load(self.MODEL_STRUCTURE_FILE, self.MODEL_WEIGHTS_FILE)
        self._frame_cnt_required = frame_cnt_required

    def predict(self, train_images: List[np.ndarray], batch_size=None) -> List[object]:
        x = np.asarray(train_images)
        x = x.reshape(x.shape[0], -1)
        predicted = self._model.predict(x, batch_size)

        return predicted
