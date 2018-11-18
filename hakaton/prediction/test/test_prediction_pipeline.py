from typing import List
from unittest import TestCase
import os
import shutil as sh

import numpy as np

from hakaton.prediction.model import SkyhacksModel
from hakaton.prediction.model_loader import ModelLoader
from hakaton.prediction.pipeline import PredictionPipeline
import random


class _DummyModel(SkyhacksModel):

    def __init__(self, predictions_func):
        self._predictions_func = predictions_func

    def predict(self, train_images: List[np.ndarray]) -> List[object]:
        return [self._predictions_func(img) for img in train_images]


class _FakeDatasetLoader(ModelLoader):

    def __init__(self, model: _DummyModel):
        self._model = model

    def load(self) -> SkyhacksModel:
        return self._model


class TestPredictionPipeline(TestCase):

    TEST_IMAGES_DIR = 'hakaton/prediction/test/test_images'


    @classmethod
    def setUpClass(cls):
        dir = cls._get_output_directory()
        sh.rmtree(dir, ignore_errors=True)
        os.makedirs(os.path.join(dir, '_output'), exist_ok=False)

    def test_if_prediction_pipeline_works(self):

        uic_presence_model = _DummyModel(
            lambda img: 0 if random.random() > .5 else 1)
        uic_label_model = _DummyModel(
            lambda img: None if random.random() > .9 else '32-24-521152-23')
        wagon_model = _DummyModel(
            lambda img: random.randint(0, 10))

        sut = PredictionPipeline(
            team_name='nienasycone_gradienty',
            data_root_directory=self.TEST_IMAGES_DIR,
            uic_presence_model_loader=_FakeDatasetLoader(uic_presence_model),
            wagon_number_model_loader=_FakeDatasetLoader(wagon_model),
            uic_label_model_loader=_FakeDatasetLoader(uic_label_model),
            output_directory=self._get_output_directory()
        )

        sut.predict()

        self.assertTrue(
            os.path.exists(os.path.join(self._get_output_directory(), 'submission.csv'))
        )

    @classmethod
    def tearDownClass(cls):
        dir = cls._get_output_directory()
        sh.rmtree(dir, ignore_errors=True)

    @staticmethod
    def _get_output_directory():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_output')

