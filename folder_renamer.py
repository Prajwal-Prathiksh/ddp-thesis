import os
import shutil
from pprint import pprint

from automan_helper import get_immediate_subdirectories

def main():
    parent_dir = os.path.join(
        os.getcwd(), 'outputs', 'tgv_2d_integrator_comparison'
    )
    from_dirs = get_immediate_subdirectories(parent_dir)
    
    for from_dir in from_dirs:
        b_from_dir = os.path.basename(from_dir)
        b_from_dir = b_from_dir.split('_re')
        b_from_dir = b_from_dir[0] + '_pst_10_re' + b_from_dir[1]
        to_dir = os.path.join(parent_dir, b_from_dir)

        print(f"from_dir: {from_dir} -> to_dir: {to_dir}")

        if not os.path.exists(to_dir):
            os.makedirs(to_dir)

        for f in os.listdir(from_dir):
            fname = os.path.join(from_dir, f)
            shutil.move(fname, to_dir)

        # Delete the old directory
        shutil.rmtree(from_dir)



if __name__ == '__main__':
    main()