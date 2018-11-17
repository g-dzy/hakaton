from functools import reduce

from hakaton.model.image import Image
from hakaton.preprocessing.resize import Resize
import glob
import os


if __name__ == '__main__':

    resize = Resize(640, 512)

    files = list(
        reduce(lambda a, b: list(a) + list(b),
               [glob.iglob(os.path.join(directory, "**", "*.jpg"), recursive=True) for directory in ['datasets/Training', 'datasets/Validation']])
    )

    output_directory_base = 'ds640x512'
    os.makedirs(output_directory_base, exist_ok=False)

    for file in files:
        image = Image(file, grayscale=True)
        image.apply_transformation(resize)
        output_path = os.path.join(output_directory_base, file)
        image.save(output_path)
