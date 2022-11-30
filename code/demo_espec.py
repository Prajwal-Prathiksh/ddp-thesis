from compyle.api import Elementwise, annotate, wrap, get_config, declare
import numpy as np
from compyle.low_level import cast
from compyle.parallel import elementwise
from compyle.config import use_config
from math import sqrt, ceil, floor
from energy_spectrum import EnergySpectrum
import inspect


@elementwise
@annotate
def compyle_1d_helper_inf_norm(i, x, y, center_x):
    j = declare('int')
    j = cast(abs(i - center_x), "int")
    y[j] += x[i]

@elementwise
@annotate(int='i, center_x', doublep='x, y')
def compyle_1d_helper_inf_norm_explicit_annotate(i, x, y, center_x):
    j = declare('int')
    j = cast(abs(i - center_x), "int")
    y[j] += x[i]

@elementwise
@annotate
def compyle_1d_helper_2_norm(i, x, y, center_x):
    j = declare('int')
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
    j = declare('int')
    tmp = sqrt((i-center_x)**2)
    frac_tmp = tmp - floor(tmp)
    if frac_tmp < 0.5:
        j = cast(floor(tmp), "int")
    else:
        j = cast(ceil(tmp), "int")
    # j = cast(round(tmp), "int") #DOESN'T WORK
    y[j] += x[i]

@elementwise
@annotate
def compyle_ek_2d_helper(
    i, ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
):
    j, wn_x, wn_y, wn = declare('int', 4)
    for j in range(box_side_y):
        wn_x = cast(abs(i - center_x), "int")
        wn_y = cast(abs(j - center_y), "int")
        wn = cast(max(wn_x, wn_y), "int")
        ek_u_sphere[wn] += ek_u[i*box_side_y + j]
        ek_v_sphere[wn] += ek_v[i*box_side_y + j]
        # Hello

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


def main_1d(func_idx:int):
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

def make_data_multi_dim(dim):
    es_ob = EnergySpectrum.from_example(dim=dim, nx=10, custom_formula=None)
    es_ob.compute(order=np.inf)
    ek_u = es_ob.ek_u
    ek_v = es_ob.ek_v
    ek_w = es_ob.ek_w
    dim = len(np.shape(ek_u))

    box_side_x = np.shape(ek_u)[0]
    box_side_y = np.shape(ek_u)[1] if dim > 1 else 0
    box_side_z = np.shape(ek_u)[2] if dim > 2 else 0

    tmp = np.array([box_side_x, box_side_y, box_side_z], dtype=np.float64)
    box_radius = int(1 + np.ceil(np.linalg.norm(tmp) / 2))

    center_x = int(box_side_x / 2)
    center_y = int(box_side_y / 2)
    center_z = int(box_side_z / 2)

    eps = 1e-50  # To avoid division by zero
    ek_u_sphere = np.zeros((box_radius, )) + eps
    ek_v_sphere = np.zeros((box_radius, )) + eps
    ek_w_sphere = np.zeros((box_radius, )) + eps

    ek = es_ob.ek

    return ek, ek_u, ek_v, ek_w, box_side_x, box_side_y, box_side_z, center_x, center_y, center_z, ek_u_sphere, ek_v_sphere, ek_w_sphere

def main_multi_dim(func_idx:int, dim:int):
    ek, ek_u, ek_v, ek_w, box_side_x, box_side_y, box_side_z, center_x, center_y, center_z, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
        make_data_multi_dim(dim=dim)
    
    with use_config(use_openmp=True):
        ek_u, ek_v, ek_u_sphere, ek_v_sphere = wrap(
            ek_u.ravel(), ek_v.ravel(), ek_u_sphere.ravel(),
            ek_v_sphere.ravel()
        )
        if dim == 2:
            func = compyle_ek_2d_helper
            func(ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y)

        # Print function definition
        bord = "-"*80
        print(bord)
        print(f"<<< Function definition for {func.__name__} >>>")
        print(inspect.getsource(func))
        print(bord)

        # Print the function source
        print(bord)
        print(f"<<< Function source for {func.__name__} >>>")
        print(func.all_source)
        print(bord)
        

        ek_u_sphere.pull()
        ek_v_sphere.pull()
    ek_actual = (ek_u_sphere.data + ek_v_sphere.data)/2
    
    print(len(ek))
    print(f"{ek = }")
    print(len(ek_actual))
    print(f"{ek_actual = }")
    error = np.max(np.abs(ek_actual - ek))
    print(f"Max error: {error}")


def main(func_idx:int, dim:int):
    if dim == 1:
        main_1d(func_idx)
    elif dim == 2:
        main_multi_dim(func_idx=func_idx, dim=dim)
    else:
        raise NotImplementedError
    bord = '*'*35
    # Print run success in green color
    print(f"\033[92m{bord} RUN SUCCESSFUL {bord}\033[0m")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    FUNC_IDX_CHOICES = [0, 1, 2, 3]
    DIM_CHOICES = [1, 2]
    parser.add_argument(
        "-f", "--func_idx", type=int, choices=FUNC_IDX_CHOICES, default=0
    )
    parser.add_argument(
        "-d", "--dim", type=int, choices=DIM_CHOICES, default=1
    )
    args = parser.parse_args()
    main(func_idx=args.func_idx, dim=args.dim)
