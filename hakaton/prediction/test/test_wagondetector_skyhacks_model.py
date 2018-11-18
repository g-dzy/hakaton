from unittest import TestCase

from hakaton.img.img_reader import ImgReader
from hakaton.prediction.wagondetector_skyhacks_model import WagonDetectorSkyhacksModel


class TestWagondetectorSkyhacksModel(TestCase):
    def test_load_model(self):
        model = WagonDetectorSkyhacksModel()
        img_reader = ImgReader()
        x = img_reader.read("hakaton/prediction/test/0_0_left_18.jpg")
        predicted = model.predict([x])
        self.assertEqual(len(x), len(predicted))
