# Library imports.
import os
import argparse

CACHE_FOLDER_NAMES = [
    "__pycache__",
    ".automan",
    
]

def get_cached_folders(path:str, recursive:bool=False, outputs:bool=False):
    """
        Get all cached folders.
    """
    if path == ".":
        path = os.getcwd()

    if outputs:
        # Warn user and get confirmation.
        print("WARNING: This will delete all outputs!")
        print("Are you sure you want to continue? (y/n)")
        if input() != "y":
            return
        CACHE_FOLDER_NAMES.append("outputs")
    
    if recursive:
        parent_dirs = []
        for root, dirs, _ in os.walk(path):
            for d in dirs:
                if d in CACHE_FOLDER_NAMES:
                    parent_dirs.append(os.path.join(root, d))
        
        return parent_dirs
    else:
        return [
            os.path.join(path, d)
            for d in CACHE_FOLDER_NAMES
            if os.path.isdir(os.path.join(path, d))
        ]

def delete_dirs(dirs:list):
    """
        Delete all directories, which are passed as a list.
        Directories are deleted whether they are empty or not.
    """
    if dirs:
        for d in dirs:
            # Check if the directory exists.
            if os.path.isdir(d):
                # Check if the directory is empty.
                if len(os.listdir(d)) == 0:
                    os.rmdir(d)
                else:
                    # Delete all files in the directory.
                    for f in os.listdir(d):
                        os.remove(os.path.join(d, f))
                    # Delete the directory.
                    os.rmdir(d)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Clean cached folders.'
    )
    parser.add_argument(
        '-r', '--recursive', action='store_true',
        help='Recursively clean cached folders.'
    )
    parser.add_argument(
        '-p', '--path', type=str, default='.', dest='path',
        help='Path to clean cached folders.'
    )
    parser.add_argument(
        '-o', '--outputs', action='store_true', dest='outputs',
        help='Clean outputs folder.'
    )
    args = parser.parse_args()

    cached_folders = get_cached_folders(
        args.path, args.recursive, args.outputs
    )
    delete_dirs(cached_folders)