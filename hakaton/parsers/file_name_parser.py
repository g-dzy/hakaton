import os


class FileMeta:

    def __init__(self, file_name: str, train_number: int, train_orientation: str, frame_number: int):
        self.file_name = file_name
        self.train_number = train_number
        self.train_orientation = train_orientation
        self.frame_number = frame_number


class FileNameParser:

    SEPARATOR = '_'
    TRAIN_NUMBER_POSITION = 1
    TRAIN_ORIENTATION_POSITION = 2
    FRAME_NUMBER_POSITION = 3

    def parse(self, file_path: str):
        file_name_with_extension = os.path.basename(file_path)
        file_name = str(file_name_with_extension.split('.')[0])
        meta = file_name.split(self.SEPARATOR)
        train_number = int(meta[self.TRAIN_NUMBER_POSITION])
        train_orientation = str(meta[self.TRAIN_ORIENTATION_POSITION])
        frame_number = int(meta[self.FRAME_NUMBER_POSITION])
        return FileMeta(
            file_path, train_number, train_orientation, frame_number
        )
