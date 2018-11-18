import os

from tensorflow.keras import models, Model


def load(model_file: str, weights_path: str) -> Model:
    with open(model_file, 'r') as json_file:
        loaded_model_json = json_file.read()
        __validate_file_exist(model_file)
        loaded_next_wagon_model = models.model_from_json(loaded_model_json)
        __validate_file_exist(weights_path)
        loaded_next_wagon_model.load_weights(weights_path)


def __validate_file_exist(file: str):
    if not os.path.isfile(file):
        FileNotFoundError(f"File '{file}' not found.")
