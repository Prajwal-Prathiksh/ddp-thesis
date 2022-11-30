from compyle.api import Elementwise, annotate, wrap, get_config, declare
import numpy as np
from compyle.low_level import cast
from compyle.parallel import elementwise
from compyle.config import use_config
from math import sqrt, ceil, floor
import inspect


@elementwise
@annotate
def compyle_1d_helper_inf_norm(i, x, y, center_x):
    j = cast(abs(i - center_x), "int")
    y[j] += x[i]

@elementwise
@annotate(int='i, center_x', doublep='x, y')
def compyle_1d_helper_inf_norm_explicit_annotate(i, x, y, center_x):
    j = cast(abs(i - center_x), "int")
    y[j] += x[i]

@elementwise
@annotate
def compyle_1d_helper_2_norm(i, x, y, center_x):
    tmp = sqrt((i-center_x)**2)
    frac_tmp = tmp - floor(tmp)
    if frac_tmp < 0.5:
        j = cast(floor(tmp), "int")
    else:
        j = cast(ceil(tmp), "int")
    # j = cast(round(tmp), "int") #DOESN'T WORK
    y[j] += x[i]

@elementwise
@annotate(int='i, center_x', doublep='x, y')
def compyle_1d_helper_2_norm_explicit_annotate(i, x, y, center_x):
    tmp = sqrt((i-center_x)**2)
    frac_tmp = tmp - floor(tmp)
    if frac_tmp < 0.5:
        j = cast(floor(tmp), "int")
    else:
        j = cast(ceil(tmp), "int")
    # j = cast(round(tmp), "int") #DOESN'T WORK
    y[j] += x[i]

def make_data():
    EPS = 1e-50
    x = np.zeros(10)
    x[4] = x[6] = 0.125

    tmp = len(x)
    box_radius = int(1 + np.ceil(tmp / 2))
    center_x = int(len(x)/2)
    y = np.zeros(box_radius) + EPS

    y_expected = np.zeros(box_radius) + EPS
    y_expected[1] += 0.125
    
    return x, center_x, y, y_expected

def main(func_idx:int):
    x, center_x, y, y_expected = make_data()

    with use_config(use_openmp=True):
        x, y = wrap(x, y)
        if func_idx == 0:
            # WORKS
            func = compyle_1d_helper_inf_norm
            # compyle_1d_helper_inf_norm(x, y, center_x)
        elif func_idx == 1:
            # DOESN'T WORK
            func = compyle_1d_helper_inf_norm_explicit_annotate
            # compyle_1d_helper_inf_norm_explicit_annotate(x, y, center_x) 
        elif func_idx == 2:
            # WORKS
            func = compyle_1d_helper_2_norm
            # compyle_1d_helper_2_norm(x, y, center_x)
        elif func_idx == 3:
            # DOESN'T WORK
            func = compyle_1d_helper_2_norm_explicit_annotate
            # compyle_1d_helper_2_norm_explicit_annotate(x, y, center_x)
        
        # Print function definition
        bord = "-"*80
        print(bord)
        print(f"<<< Function definition for {func.__name__} >>>")
        print(inspect.getsource(func))
        print(bord)

        func(x, y, center_x)

        # Print the function source
        print(bord)
        print(f"<<< Function source for {func.__name__} >>>")
        print(func.source)
        print(bord)


        y.pull()

    y_actual = y.data/2
    error = np.max(np.abs(y_actual - y_expected))
    print(f"Max error: {error}")
    bord = '*'*30
    print(f"{bord} RUN SUCCESSFUL {bord}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    FUNC_IDX_CHOICES = [0, 1, 2, 3]
    parser.add_argument(
        "-f", "--func_idx", type=int, choices=FUNC_IDX_CHOICES, default=0
    )
    args = parser.parse_args()
    main(args.func_idx)
