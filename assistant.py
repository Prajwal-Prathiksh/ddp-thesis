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
            ln = line.strip()
            if ln.startswith("def ") or ln.startswith("class "):
                # Find the next line which ends with a colon.
                j = i
                while not lines[j].endswith(":\n"):
                    j += 1
                next_line = lines[j + 1].strip()
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


def run_autopep8(fpath, in_place=True):
    """
    Runs autopep8 on the file at fpath.

    Parameters
    ----------
    fpath : str
        The path to the file to run autopep8 on.
    in_place : bool
        Whether to run autopep8 in place or not.
    """
    # If only base name is provided, get the full path.
    if os.path.basename(fpath) == fpath:
        fpath = os.path.join(os.getcwd(), fpath)
    print(fpath)
    cmd = f'autopep8 "{fpath}" --aggressive --aggressive --verbose'
    if in_place:
        cmd += " --in-place"
    else:
        cmd += " --diff"
    print(cmd)
    os.system(cmd)


def cli_args():
    """
    Returns the command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='An assistant to help with code quality, & other things.'
        '\nNote: To ignore a file, add "#NOQA" to the first line of the file.',
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
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Run a specified check even if the file has a '#NOQA' comment."
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

    # If no checks are specified, run all checks.
    if not (check_docstrings or check_todos or run_pep8):
        check_docstrings = True
        check_todos = True
        run_pep8 = True

    # Take user input to run autopep8 as diff
    if run_pep8:
        in_place = input(
            "Run autopep8 as diff instead of in place? (Y/n): "
        ).lower() == "n"
    else:
        in_place = False

    n_border = 70
    start_border = "<" * (n_border // 4) + "-" * (n_border // 4) * 3
    end_border = "-" * (n_border // 4) * 3 + ">" * (n_border // 4)

    found_missing_docstrings = False
    found_todos = False
    for fpath in py_files:
        with open(fpath, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if "#NOQA" in first_line and not args.force:
                continue
        if check_docstrings:
            print(start_border)
            print("Docstrings:")
            print(f"[file: .\\{os.path.basename(fpath)}]")
            found_missing_docstrings = (
                find_missing_doctrings(cwd, fpath) or found_missing_docstrings
            )
            if not found_missing_docstrings:
                print("\033[33mNo missing docstrings found.\033[0m")
            print(end_border)
        if check_todos:
            print(start_border)
            print("TODOs:")
            print(f"[file: .\\{os.path.basename(fpath)}]")
            found_todos = find_todos(cwd, fpath) or found_todos
            if not found_todos:
                print("\033[33mNo TODOs found.\033[0m")
            print(end_border)
        if run_pep8:
            print(start_border)
            print("PEP8:")
            run_autopep8(fpath, in_place)
            print(end_border)
        print("\n")


if __name__ == "__main__":
    main()
