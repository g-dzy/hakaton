from typing import List

from tensorflow.keras import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout


class WagonDetectorModel:
    def __init__(self, metrics: List[str] = None):
        self._metrics = metrics or ["accuracy"]
        self._model_structure = None

    def model(self):
        if self._model_structure:
            return self._model_structure
        self._model_structure = self._build_model()

        return self._model_structure

    def _build_model(self):
        model_wagon_detector = Sequential()

        model_wagon_detector.add(Dense(64, input_shape=(256 * 320,)))
        model_wagon_detector.add(Dropout(0.4))
        model_wagon_detector.add(BatchNormalization())
        model_wagon_detector.add(LeakyReLU())

        model_wagon_detector.add(Dense(64))
        model_wagon_detector.add(Dropout(0.4))
        model_wagon_detector.add(BatchNormalization())
        model_wagon_detector.add(LeakyReLU())

        model_wagon_detector.add(Dense(2, activation="softmax"))

        optimizer = optimizers.Adam(lr=0.001)
        model_wagon_detector.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=self._metrics)

        return model_wagon_detector
