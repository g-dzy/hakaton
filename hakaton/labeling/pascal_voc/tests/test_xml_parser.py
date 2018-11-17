from unittest import TestCase

from hakaton.labeling.pascal_voc.pascal_voc_parser import PascalVocParser


class TestXmlParser(TestCase):

    def test_if_empty_image_is_parsed(self):

        path = 'hakaton/labeling/pascal_voc/samples/empty_1.xml'
        sut = PascalVocParser()
        result = sut.parse(path)

        self.assertEqual(result.file_path, 'empty_1_path')
        self.assertEqual(len(result.boxes), 0)

    def test_if_single_object_image_is_parsed(self):

        path = 'hakaton/labeling/pascal_voc/samples/single_1.xml'
        sut = PascalVocParser()
        result = sut.parse(path)

        self.assertEqual(result.file_path, 'single_1_path')
        self.assertEqual(len(result.boxes), 1)
        self.assertEqual(result.boxes[0].label, 1)
        self.assertEqual(result.boxes[0].xmin, 5)
        self.assertEqual(result.boxes[0].ymin, 73)
        self.assertEqual(result.boxes[0].xmax, 114)
        self.assertEqual(result.boxes[0].ymax, 140)

    def test_if_multiple_objects_image_is_parsed(self):

        path = 'hakaton/labeling/pascal_voc/samples/multiple_2.xml'
        sut = PascalVocParser()
        result = sut.parse(path)

        self.assertEqual(result.file_path, 'multiple_2_path.jpg')

        self.assertEqual(len(result.boxes), 2)

        self.assertEqual(result.boxes[0].label, 3)
        self.assertEqual(result.boxes[0].xmin, 134)
        self.assertEqual(result.boxes[0].ymin, 100)
        self.assertEqual(result.boxes[0].xmax, 148)
        self.assertEqual(result.boxes[0].ymax, 126)

        self.assertEqual(result.boxes[1].label, 2)
        self.assertEqual(result.boxes[1].xmin, 149)
        self.assertEqual(result.boxes[1].ymin, 99)
        self.assertEqual(result.boxes[1].xmax, 163)
        self.assertEqual(result.boxes[1].ymax, 127)

