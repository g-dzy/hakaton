from typing import List


class ObjectDetectionLabel:

    class Object:

        def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int, label= None):
            self.xmin = xmin
            self.xmax = xmax
            self.ymin = ymin
            self.ymax = ymax
            try:
                self.label = int(label)
            except:
                self.label = label

    def __init__(self,
                 file_path: str,
                 boxes: List[Object]
                 ):

        self.file_path = file_path
        self.boxes = boxes
