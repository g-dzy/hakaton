from unittest import TestCase

from hakaton.corpus.corpus import Corpus
from hakaton.img.img_reader import ImgReader


class TestCorpus(TestCase):
    TRAINING_DIR = "dataset/Internal"
    METADATA_FILE_PATH = "dataset/Internal/0_0_0_metadata_extended.csv"

    @classmethod
    def setUpClass(cls):
        cls._corpus = Corpus(ImgReader(), cls.METADATA_FILE_PATH, cls.TRAINING_DIR)

    def test_get_by_type(self):
        imgs = self._corpus.get_by_type(0)
        self.assertEqual(imgs.shape, (5, 1024, 1280, 3))

        imgs = self._corpus.get_by_type(1)
        self.assertEqual(imgs.shape, (4, 1024, 1280, 3))

    def test_get_by_uic(self):
        imgs = self._corpus.get_by_uic(0)
        self.assertEqual(imgs.shape, (4, 1024, 1280, 3))

        imgs = self._corpus.get_by_uic(1)
        self.assertEqual(imgs.shape, (5, 1024, 1280, 3))
