import os
import shutil
from typing import List


def join_to_file_dir_path(file, path):
    return os.path.join(os.path.dirname(file), path)


def list_dir_files(dir: str) -> List[str]:
    return [os.path.join(dir, f) for f in os.listdir(dir)]


def list_dir_files_recursively(root_dir: str) -> List[str]:
    files = list()
    for single_file in os.listdir(root_dir):
        single_file = os.path.join(root_dir, single_file)
        if os.path.isdir(single_file):
            dir_files = list_dir_files_recursively(single_file)
            files.extend(dir_files)
            continue
        files.append(single_file)

    return files


def copy_files_recursively(root_dir: str, dest: str):
    files_to_copy = list_dir_files_recursively(root_dir)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    for file in files_to_copy:
        shutil.copy(file, dest)
