import glob

from hakaton.labeling.pascal_voc.csv_writer import PascalVocWriter
from hakaton.labeling.pascal_voc.pascal_voc_parser import PascalVocParser


if __name__ == '__main__':

    input_directory_pattern = 'datasets/dawidowe/__MACOSX/Training/**/*.xml'
    output_csv_path = 'datasets\dawidowe\labels.csv'

    # input_directory_pattern = 'datasets\ds320x256\labels\**\*.xml'
    # output_csv_path = 'datasets\ds320x256\labels.csv'

    parser = PascalVocParser()
    writer = PascalVocWriter()

    for filename in glob.iglob(input_directory_pattern, recursive=True):
        obj = parser.parse(filename)
        writer.append(obj)

    writer.to_csv(output_csv_path)
