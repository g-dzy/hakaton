from unittest import TestCase

from hakaton.model.image import Image


class TestImage(TestCase):

    SAMPLE_PATH = 'sample_image.jpg'

    def test_if_image_is_read(self):

        img = Image(self.SAMPLE_PATH)

        self.assertIsNotNone(img.data)
        self.assertEqual(img.size, (720, 602))
        self.assertEqual(img.width, 602)
        self.assertEqual(img.height, 720)

    def test_if_is_grayscale(self):

        img = Image(self.SAMPLE_PATH, grayscale=True)
        self.assertEqual(len(img.data.shape), 2)

    def test_if_is_rgb(self):

        img = Image(self.SAMPLE_PATH, grayscale=False)
        self.assertEqual(len(img.data.shape), 3)
