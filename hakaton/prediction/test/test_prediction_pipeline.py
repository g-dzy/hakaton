from unittest import TestCase
import os
import shutil as sh


class TestPredictionPipeline(TestCase):

    TEST_IMAGES_DIR = 'hakaton/prediction/test/test_images'


    @classmethod
    def setUpClass(cls):
        dir = cls._get_output_directory()
        sh.rmtree(dir, ignore_errors=True)
        os.makedirs(os.path.join(dir, '_output'), exist_ok=False)

    def test_if_prediction_pipeline_works(self):

        # TODO implement
        self.assertTrue(False)

    @classmethod
    def tearDownClass(cls):
        dir = cls._get_output_directory()
        sh.rmtree(dir, ignore_errors=True)

    @staticmethod
    def _get_output_directory():
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_output')

