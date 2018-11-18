from typing import List

from tensorflow.keras import optimizers
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, BatchNormalization, Dropout, Activation


class UicDetectorModel:
    def __init__(self, metrics: List[str]):
        self._metrics = metrics or list()
        self._model_structure = None

    def model(self):
        if self._model_structure:
            return self._model_structure
        self._model_structure = self._build_model()

        return self._model_structure

    def _build_model(self):
        metrics = ["accuracy"]

        model_uic = Sequential()

        model_uic.add(Dense(32, input_shape=(256 * 320,)))
        model_uic.add(Dropout(0.4))
        model_uic.add(BatchNormalization())
        model_uic.add(Activation("tanh"))

        model_uic.add(Dense(32))
        model_uic.add(Dropout(0.4))
        model_uic.add(BatchNormalization())
        model_uic.add(Activation("tanh"))

        model_uic.add(Dense(2, activation="softmax"))

        optimizer = optimizers.Adam(lr=0.0004)
        model_uic.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

        return model_uic
