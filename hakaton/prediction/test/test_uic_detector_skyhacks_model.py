from unittest import TestCase

from hakaton.img.img_reader import ImgReader
from hakaton.prediction.uicdetector_skyhacks_model import UicDetectorSkyhacksModel


class TestUicDetectorShyhacksModel(TestCase):
    def test_load_model(self):
        model = UicDetectorSkyhacksModel()
        img_reader = ImgReader()
        x = img_reader.read("hakaton/prediction/test/0_0_left_18.jpg")
        predicted = model.predict([x])
        self.assertEqual(len(x), len(predicted))
