r"""
An `automan` assistant.
###
"""
# Libray imports
import os
import sys
import shutil
import json
import yaml
import warnings
import argparse

# Define global variables
YAML_FNAME = None


def get_immediate_subdirectories(dir):
    """
    Get all immediate subdirectories in a directory.

    Parameters
    ----------
    dir : str
        The directory to get subdirectories from.

    Returns
    -------
    subdirs : list
        A list of all subdirectories in the directory.
    """
    dir = os.path.abspath(dir)
    subdirs = []
    for name in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, name)):
            subdirs.append(os.path.join(dir, name))
    return subdirs


def human_readable_size(size, precision=2):
    """
    Convert a size in bytes to a human readable form.

    Parameters
    ----------
    size : float
        The size in bytes.
    precision : int, optional
        The number of decimal places to round to.

    Returns
    -------
    size : str
        The size in human readable form.
    """
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    suffix_index = 0
    while size > 1024 and suffix_index < 4:
        suffix_index += 1
        size = size / 1024.0
    return "%.*f%s" % (precision, size, suffixes[suffix_index])


def calculate_dir_size(dir, human_readable=True):
    """
    Calculate the size of a directory.

    Parameters
    ----------
    dir : str
        The directory to calculate the size of.
    human_readable : bool, optional
        If True, the size is returned in human readable form as a string,
        otherwise it is returned as a float in bytes.

    Returns
    -------
    size : str/float
        The size of the directory.
    """
    dir = os.path.abspath(dir)
    size = 0
    for root, _, files in os.walk(dir):
        for f in files:
            f = os.path.join(root, f)
            size += os.path.getsize(f)
    if human_readable:
        size = human_readable_size(size)
    return size


def read_job_info(dir):
    """
    Read the job_info.json file in a directory.

    Parameters
    ----------
    dir : str
        The directory to read the job_info.json file from.

    Returns
    -------
    job_info : dict
        The job_info.json file as a dictionary.
    """
    dir = os.path.abspath(dir)
    fname = os.path.join(dir, "job_info.json")
    if not os.path.isfile(fname):
        # Raise warning in yellow
        warnings.warn("\033[93m{}\033[00m" .format(
            "job_info.json not found in {}. Skipping directory.".format(dir)))
        return None
    with open(fname, "r") as f:
        job_info = json.load(f)
    return job_info


def categorise_jobs(subdirs):
    """
    Categorise jobs into complete, incomplete, and errored, given a list of
    directories.

    Parameters
    ----------
    subdirs : list
        A list of directories to search for incomplete jobs.

    Returns
    -------
    categories : dict
        A dictionary of categories and the directories in each category.
        Keys are "done", "running", and "error".
    """
    categories = {"done": [], "running": [], "error": []}
    ids = list(categories.keys())
    for d in subdirs:
        job_info = read_job_info(d)

        if job_info is None:
            subdirs.remove(d)
            continue

        for id in ids:
            if job_info['status'] == id:
                categories[id].append(d)
                break
    return categories


def create_yaml_file(dir=None):
    """
    Create the summary.yaml file containing the summary of `automan` output
    directories.

    Parameters
    ----------
    dir : str, optional
        The directory to create the summary.yaml file in. If None, the current
        working directory is used.
    """
    global YAML_FNAME
    if dir is None:
        dir = os.getcwd()
    dir = os.path.abspath(dir)
    fname = os.path.join(dir, ".automan", "summary.yaml")
    if os.path.isfile(fname):
        YAML_FNAME = fname
        return

    # Check if .automan directory exists
    if not os.path.isdir(os.path.join(dir, ".automan")):
        os.mkdir(os.path.join(dir, ".automan"))
    with open(fname, "w") as f:
        f.write("")
    print("Created summary.yaml file in {}.".format(dir))
    YAML_FNAME = fname


def read_yaml_file():
    """
    Read the summary.yaml file.

    Returns
    -------
    summary : dict
        The summary.yaml file as a dictionary.
    """
    if YAML_FNAME is None:
        return None
    with open(YAML_FNAME, "r") as f:
        summary = yaml.safe_load(f)
    if summary is None:
        summary = {}
    return summary


def write_dir_summary(dir, size, categories):
    """
    Write the summary of a directory to the summary.yaml file.

    Parameters
    ----------
    dir : str
        The directory to write the summary of.
    size : str
        The size of the directory.
    categories : dict
        The categories of the directory.
    """
    if YAML_FNAME is None:
        return
    dir = os.path.basename(os.path.abspath(dir))
    dir_data_dict = {dir:dict(size=size)}
    for key, value in categories.items():
        dir_data_dict[dir][key] = dict(
            count=len(value), dirs=[os.path.basename(d) for d in value]
        )
    
    # Read summary.yaml
    summary = read_yaml_file()

    # Check if dir already exists in summary.yaml and update if it does
    if dir in summary:
        summary[dir].update(dir_data_dict[dir])
    else:
        summary.update(dir_data_dict)

    # Dump dir_data_dict into summary.yaml
    with open(YAML_FNAME, "w") as f:
        yaml.dump(summary, f, default_flow_style=False)

def cli_args():
    """
    Returns the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='An assistant to help with output dirs created through '
        ' `automan`.',        
        epilog="Example: "
        "\n\t>>> python automan_helper.py [options] dir1 dir2 dir3"
        "\n\t>>> python automan_helper.py -od outpts -d done",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "dirs", nargs="*", help="The directories to clean out.",
    )
    parser.add_argument(
        "-o", "--output-dir", type=str, dest="output_dir",
        help="The output directory containing all of `automan`'s output "
        "directories. If specified, the script will search for all "
        "subdirectories in this directory and use them as input directories."
    )
    parser.add_argument(
        "-d", "--delete", type=str, dest="delete", default="none",
        choices=["none", "done", "running", "error"],
        help="Delete directories in the specified category."
    )
    parser.add_argument(
        "-s", "--save-yaml", action="store_true", dest="save_yaml",
        help="Save a .yaml file containing the summary of `automan` output "
        "directories."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose",
        help="Print more information."
    )

    return parser.parse_args()


def main(dirs, delete=None, save_yaml=True, verbose=False):
    """
    Main function.
    """
    # Colors for printing
    colors = dict(
        done="\033[92m",
        running="\033[93m",
        error="\033[91m",
    )

    if save_yaml:
        create_yaml_file(dir=os.getcwd())

    for dir in dirs:
        subdirs = get_immediate_subdirectories(dir)
        dir_size = calculate_dir_size(dir, human_readable=True)
        categories = categorise_jobs(subdirs)
        print("Directory: {}".format(dir))
        print("Size: {}".format(dir_size))

        if save_yaml:
            write_dir_summary(dir=dir, size=dir_size, categories=categories)
        
        for key in categories:
            print(
                "{}  {}: {}{}".format(
                    colors[key], key.title(),
                    len(categories[key]), "\033[00m"
                )
            )
            if verbose:
                for d in categories[key]:
                    print(
                        "{}    {}{}".format(
                            colors[key], os.path.basename(d), "\033[00m"
                        )
                    )

        if delete is not None:
            print(
                "Deleting{} {} {}directories...".format(
                    colors[delete], delete.title(), "\033[00m"
                )
            )
            confirm = input("Are you sure? (y/n): ")
            if confirm.lower() == "n":
                print("Not deleting.")
            else:
                print("Deleting...")
                for d in categories[delete]:
                    shutil.rmtree(d)
                print("Done deleting.")


if __name__ == "__main__":
    args = cli_args()
    if args.dirs == [] and args.output_dir is None:
        # Print in red
        print("\033[91m{}\033[00m" .format("No directories provided."))
        sys.exit(1)

    if args.output_dir is not None:
        args.dirs = []
        args.dirs.extend(get_immediate_subdirectories(args.output_dir))
    delete = None
    if args.delete != "none":
        delete = args.delete

    main(
        dirs=args.dirs, delete=delete, save_yaml=args.save_yaml, verbose=args.verbose
    )
