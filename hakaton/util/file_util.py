import os
from typing import List


def join_to_file_dir_path(file, path):
    return os.path.join(os.path.dirname(file), path)


def list_dir_files(dir: str) -> List[str]:
    return [os.path.join(dir, f) for f in os.listdir(dir)]
