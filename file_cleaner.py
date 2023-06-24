import os
import shutil
from pprint import pprint
from automan_helper import get_immediate_subdirectories

UNWANTED_EXTENSIONS = [
    '.npz', '.json', '.py', '.csv', '.txt', '.hdf5', 'info'
]

def get_all_files_in_dir(dir):
    files = []
    for root, _, filenames in os.walk(dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def main():
    parent_dir = os.path.join(
        os.getcwd(), 'report', 'LaTeX Files', 'Code-Figures'
    )
    files = get_all_files_in_dir(parent_dir)
    print('Total files: {}'.format(len(files)))
    files_to_remove = []
    for f in files:
        if any([f.endswith(ext) for ext in UNWANTED_EXTENSIONS]):
            files_to_remove.append(f)
    print('Files to remove: {}'.format(len(files_to_remove)))
    
    if len(files_to_remove) < 1:
        print('No files to remove.')
        return

    resp = input('Are you sure you want to remove these files? (y/N): ')
    if resp == 'y':
        for f in files_to_remove:
            os.remove(f)
        print('Removed files.')

if __name__ == '__main__':
    main()