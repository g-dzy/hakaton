from typing import List

from hakaton.labeling.pascal_voc.object_detection_label import ObjectDetectionLabel
import xml.etree.ElementTree as ET


class PascalVocParser:

    def __init__(self):
        pass

    def parse(self, file: str) -> ObjectDetectionLabel:

        tree = ET.parse(file)
        root = tree.getroot()
        path = root.find('path').text
        objects = root.findall('object')

        object_detection_label = ObjectDetectionLabel(path, boxes=self._map_boxes(objects))
        return object_detection_label

    def _map_boxes(self, objects: list) -> List[ObjectDetectionLabel.Object]:

        def _map_object(obj) -> ObjectDetectionLabel.Object:

            label = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            return ObjectDetectionLabel.Object(xmin, xmax, ymin, ymax, label)

        return list(map(_map_object, objects))
