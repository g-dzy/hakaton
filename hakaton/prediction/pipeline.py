import os
from logging import getLogger
import glob
from typing import List

from hakaton.model.image import Image
from hakaton.parsers.file_name_parser import FileNameParser, FileMeta
from hakaton.prediction.model_loader import ModelLoader
from hakaton.prediction.submission_file_builder import SubmissionFileBuilder
from hakaton.preprocessing.resize import Resize


class PredictionPipeline:

    def __init__(self,
                 team_name: str,
                 data_root_directory: str,
                 uic_presence_model_loader: ModelLoader,
                 wagon_number_model_loader: ModelLoader,
                 uic_label_model_loader: ModelLoader
                 ):

        self._logger = getLogger()
        self._logger.info('Listing directory %s contents' % data_root_directory)
        self._train_directories = [os.path.join(data_root_directory, d) for d in os.listdir(data_root_directory)]
        self._files_meta_parser = FileNameParser()
        self._resize_transformation = Resize(320, 256)
        self._submission_file_builder = SubmissionFileBuilder(team_name)

        # TODO: standard scaler if location will be detected
        self._uic_presence_model = uic_presence_model_loader.load()
        self._wagon_number_model = wagon_number_model_loader.load()
        self._uic_label_model = uic_label_model_loader.load()

    def predict(self):

        for train_directory in self._train_directories:
            train_id = os.path.basename(train_directory).split('_')[-1]
            self._logger.info('Starting pipeline for train %s' % train_id)
            self._logger.info('Listing directories of train %s' % train_id)
            files = glob.glob(train_directory, recursive=True)
            self._logger.info('Parsing files metadata of train %s' % train_id)
            files = [self._files_meta_parser.parse(f) for f in files]
            ordered_files = sorted(files, key=lambda f: f.frame_number)
            self._logger.info('Preprocessing images of train %s' % train_id)
            images = self._preprocess([o.file_name for o in ordered_files])
            ndarrays = [i.ndarray for i in images]
            self._logger.info('Predicting wagon numbers of train %s' % train_id)
            wagon_numbers = self._wagon_number_model.predict(ndarrays)
            self._logger.info('Predicting uic presence of train %s' % train_id)
            uic_code_presences = self._uic_presence_model.predict(ndarrays)
            self._logger.info('Predicting uic labels of train %s' % train_id)
            uic_code_labels = self._uic_label_model.predict(ndarrays)
            train_output = zip(files, wagon_numbers, uic_code_presences, uic_code_labels)
            self._update_submission_file(list(train_output))

    def _preprocess(self, files) -> List[Image]:

        return [
            Image(file, grayscale=True).apply_transformation(self._resize_transformation)
            for file in files
        ]

    def _update_submission_file(self, outputs: list):

        for output in outputs:

            file_meta: FileMeta = output[0]
            wagon_number: str = output[1]
            uic_code_present: int = output[2]
            uic_code_label: str = output[3]

            self._submission_file_builder.append_prediction(
                train_number=file_meta.train_number,
                left_or_right=file_meta.train_orientation,
                frame_number=file_meta.frame_number,
                wagon=wagon_number,
                uic_01=int(uic_code_present),
                uic_label=uic_code_label
            )
