import pandas as pd

from unittest import TestCase


class TestReadExtendedMetadata(TestCase):
    EXT_METADATA_FILE = "dataset/Training/0_0/0_0_0_metadata_extended.csv"

    def test_read_ext_metadata(self):
        df = pd.read_csv(self.EXT_METADATA_FILE)
        expected_columns = ["filename", "type", "is_uic", "uic_value"]
        self.assertListEqual(df.columns.values.tolist(), expected_columns)
        print(df.to_string())
