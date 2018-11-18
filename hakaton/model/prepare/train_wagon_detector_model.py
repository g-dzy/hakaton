from hakaton.corpus.columns import Columns
from hakaton.corpus.corpus import Corpus
from hakaton.img.img_reader import ImgReader
from hakaton.model.prepare.model_generator import ModelGenerator
from hakaton.model.wagon_detector_model import WagonDetectorModel

META_FILE_PATH = "dataset/transformed/320x256_flatten/Training/selected/0_0_0_metadata_extended.csv"
FILES_ROOT_DIR = "dataset/transformed/320x256_flatten/Training/selected"
OUT_DIR = "output/wagondetector"

BATCH_SIZE = 8
EPOCHS = 1
IS_SHUFFLE = True
VALIDATION_SPLIT = 0.2
TRAIN_LABEL = Columns.TYPE.value


def main():
    wagon_detector_model = WagonDetectorModel()
    model = wagon_detector_model.model()
    img_reader = ImgReader()
    corpus = Corpus(img_reader, META_FILE_PATH, FILES_ROOT_DIR)
    model_generator = ModelGenerator(model, corpus, img_reader)
    model_generator.generate(TRAIN_LABEL, OUT_DIR, IS_SHUFFLE, epochs=EPOCHS, batch_size=BATCH_SIZE,
                             validation_split=VALIDATION_SPLIT)


if __name__ == '__main__':
    main()
