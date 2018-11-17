from logging import getLogger

import pandas as pd
import os
from hakaton.labeling.pascal_voc.object_detection_label import ObjectDetectionLabel


class PascalVocWriter:

    def __init__(self, expected_objects_on_scene: int = None):

        features = ['path']
        self._logger = getLogger()

        if expected_objects_on_scene:
            features += [self._format_feature_name(i) for i in range(expected_objects_on_scene)]

        self._expected_objects_on_scene = expected_objects_on_scene

        self._df = pd.DataFrame(columns=features)

    def append(self, obj: ObjectDetectionLabel):

        if self._expected_objects_on_scene:
            if len(obj.boxes) < self._expected_objects_on_scene:
                self._logger.warning(
                    'There is fewer labeled observations than expected: %d vs %d' %
                    (len(obj.boxes), self._expected_objects_on_scene)
                )

            elif len(obj.boxes) > self._expected_objects_on_scene:
                self._logger.error(
                    'There is more labeled observations than expected: %d vs %d' %
                    (len(obj.boxes), self._expected_objects_on_scene)
                )

                raise Exception("Well, NOPE")

        row = {}

        row['path'] = obj.file_path

        for i, box in enumerate(obj.boxes):
            row[self._format_feature_name(i, 'label')] = box.label
            row[self._format_feature_name(i, 'xmin')] = box.xmin
            row[self._format_feature_name(i, 'xmax')] = box.xmax
            row[self._format_feature_name(i, 'ymin')] = box.ymin
            row[self._format_feature_name(i, 'ymax')] = box.ymax

        self._df = self._df.append(row, ignore_index=True)

    def to_csv(self, path: str):

        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        self._df.to_csv(path, index=False)

    def _format_feature_name(self, object_id: int, suffix: str = '') -> str:
        return 'object_%d_%s' % (object_id, suffix)
