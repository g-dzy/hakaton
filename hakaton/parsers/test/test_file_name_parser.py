from unittest import TestCase

from hakaton.parsers.file_name_parser import FileNameParser


class TestFileNameParser(TestCase):

    def test_simple_file_path(self):

        sut = FileNameParser()
        result = sut.parse("0_10_left_12.jpg")
        self.assertEqual(result.train_number, 10)
        self.assertEqual(result.train_orientation, 'left')
        self.assertEqual(result.frame_number, 12)

    def test_relative_path(self):
        sut = FileNameParser()
        result = sut.parse("dataset/Training/0_0/0_0_left/0_13_right_12.jpg")
        self.assertEqual(result.train_number, 13)
        self.assertEqual(result.train_orientation, 'right')
        self.assertEqual(result.frame_number, 12)

    def test_absolute_path(self):
        sut = FileNameParser()
        result = sut.parse("C://Training/0_0/0_0_left/0_13_right_12.jpg")
        self.assertEqual(result.train_number, 13)
        self.assertEqual(result.train_orientation, 'right')
        self.assertEqual(result.frame_number, 12)

