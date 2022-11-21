r"""
An assistant to help with code quality, & other things.
###
"""
# Libray imports
import os
import argparse


def get_all_py_files(path):
    """
    Returns a list of all .py files in the directory tree rooted at path.

    Parameters
    ----------
    path : str
        The root directory.

    Returns:
    --------
    py_files : list
        A list of all .py files in the directory tree rooted at path.
    """
    py_files = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                py_files.append(os.path.join(root, file))
    return py_files


def find_missing_doctrings(cwd, fpath):
    """
    Checks the docstrings in the file at fpath.

    Parameters
    ----------
    cwd : str
        The current working directory.
    fpath : str
        The path to the file to check.

    Returns
    -------
    found_missing_docstrings : bool
        True if any missing docstrings were found, False otherwise.
    """
    found_missing_docstrings = False
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("def ") or line.startswith("class "):
                next_line = lines[i + 1].strip()
                if not next_line.startswith('"""'):
                    print(f"{fpath[len(cwd) + 1:]}: {i + 1}: {line.strip()}")
                    found_missing_docstrings = True

    return found_missing_docstrings


def find_todos(cwd, fpath):
    """
    Checks the file at fpath for TODOs.

    Parameters
    ----------
    cwd : str
        The current working directory.
    fpath : str
        The path to the file to check.

    Returns
    -------
    found_todos : bool
        True if any TODOs were found, False otherwise.
    """
    found_todos = False
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "TODO" in line:
                print(f"{fpath[len(cwd) + 1:]}: {i + 1}: {line.strip()}")
                found_todos = True

    return found_todos


def run_autopep8(fpath):
    """
    Runs autopep8 on the file at fpath.

    Parameters
    ----------
    fpath : str
        The path to the file to run autopep8 on.
    """
    # If only base name is provided, get the full path.
    if os.path.basename(fpath) == fpath:
        fpath = os.path.join(os.getcwd(), fpath)
    cmd = f"autopep8 {fpath} --in-place --aggressive --aggressive --verbose"
    os.system(cmd)


def cli_args():
    """
    Returns the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='An assistant to help with code quality, & other things.',
        epilog='Example: python assistant.py -d -t'
    )

    parser.add_argument(
        "files", nargs="*", help="The files to run the code on. If not "
        "provided, all .py files in the current directory will be used."
    )
    parser.add_argument(
        "-d", "--docstrings", action="store_true",
        help="Check for missing docstrings."
    )
    parser.add_argument(
        "-t", "--todos", action="store_true",
        help="Check for TODOs."
    )
    parser.add_argument(
        "-p", "--pep8", action="store_true",
        help="Correct code to conform to PEP8 (requires autopep8)."
    )
    parser.add_argument(
        "-A", "--all", action="store_true",
        help="Run all checks."
    )
    args = parser.parse_args()
    return args


def main():
    """
    Main function.
    """
    args = cli_args()
    cwd = os.getcwd()
    if args.files:
        py_files = args.files
    else:
        py_files = get_all_py_files(os.getcwd())

    current_file = os.path.basename(__file__)
    for i, fpath in enumerate(py_files):
        if os.path.basename(fpath) == current_file:
            py_files.pop(i)
            break

    check_docstrings = args.docstrings or args.all
    check_todos = args.todos or args.all
    run_pep8 = args.pep8 or args.all

    found_missing_docstrings = False
    found_todos = False
    for fpath in py_files:
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if "#NOQA" in first_line:
                continue
        if check_docstrings:
            found_missing_docstrings = (
                find_missing_doctrings(cwd, fpath) or found_missing_docstrings
            )
        if check_todos:
            found_todos = find_todos(cwd, fpath) or found_todos
        if run_pep8:
            run_autopep8(fpath)

    if check_docstrings and not found_missing_docstrings:
        print("\nNo missing docstrings found!")
    if check_todos and not found_todos:
        print("\nNo TODOs found!")


if __name__ == "__main__":
    main()
