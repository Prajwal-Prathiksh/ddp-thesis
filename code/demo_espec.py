from compyle.api import annotate, wrap, declare
import numpy as np
from compyle.low_level import cast
from compyle.parallel import elementwise
from compyle.config import use_config
from math import sqrt, ceil, floor
from energy_spectrum import EnergySpectrum


@elementwise
@annotate
def compyle_1d_helper_inf_norm(i, x, y, center_x):
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
@annotate
def compyle_ek_2d_helper_inf_norm(
    i, ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
):
    iter_i, iter_j, wn = declare('int', 3)

    iter_i = i // box_side_y
    iter_j = i - iter_i * box_side_y

    wn = cast(max(abs(iter_i - center_x), abs(iter_j - center_y)), "int")

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]

@elementwise
@annotate
def compyle_ek_2d_helper_2_norm(
    i, ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
):
    iter_i, iter_j, wn = declare('int', 3)

    iter_i = i // box_side_y
    iter_j = i - iter_i * box_side_y

    tmp, frac_tmp, floor_tmp = declare('double', 2)

    tmp = cast(sqrt((iter_i-center_x)**2 + (iter_j-center_y)**2), 'double')
    floor_tmp = cast(floor(tmp), 'double')
    frac_tmp = tmp - floor_tmp
    if frac_tmp < 0.5:
        wn = cast(floor_tmp, "int")
    else:
        wn = cast(floor_tmp + 1.0, "int")
    
    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]

@elementwise
@annotate
def compyle_ek_3d_helper_inf_norm(
    i, ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere, box_side_y, box_side_z, center_x, center_y, center_z
):
    iter_i, iter_j, iter_k, wn = declare('int', 4)

    iter_i = i // (box_side_y * box_side_z)
    iter_j = (i - iter_i * box_side_y * box_side_z) // box_side_z
    iter_k = i - iter_i * box_side_y * box_side_z - iter_j * box_side_z

    wn = cast(max(abs(iter_i - center_x), abs(iter_j - center_y), abs(iter_k - center_z)), "int")

    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]
    ek_w_sphere[wn] += ek_w[i]

    # printf("i: %d\n", i)
    # printf("iter_i: %d, iter_j: %d, iter_k: %d\n", iter_i, iter_j, iter_k)

    # printf("wn: %d\n", wn)
    # printf("ek_u: %f, ek_v: %f, ek_w: %f\n", ek_u[i], ek_v[i], ek_w[i])
    # printf("ek_u_sp: %f, ek_v_sp: %f, ek_w_sp: %f\n\n", ek_u_sphere[wn], ek_v_sphere[wn], ek_w_sphere[wn])


@elementwise
@annotate
def compyle_ek_3d_helper_2_norm(
    i, ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere, box_side_y, box_side_z, center_x, center_y, center_z
):
    iter_i, iter_j, iter_k, wn = declare('int', 4)

    iter_i = i // (box_side_y * box_side_z)
    iter_j = (i - iter_i * box_side_y * box_side_z) // box_side_z
    iter_k = i - iter_i * box_side_y * box_side_z - iter_j * box_side_z

    tmp, frac_tmp, floor_tmp = declare('double', 3)

    tmp = cast(sqrt((iter_i-center_x)**2 + (iter_j-center_y)**2 + (iter_k-center_z)**2), 'double')
    floor_tmp = cast(floor(tmp), 'double')
    frac_tmp = tmp - floor_tmp
    if frac_tmp < 0.5:
        wn = cast(floor_tmp, "int")
    else:
        wn = cast(floor_tmp + 1.0, "int")
    
    ek_u_sphere[wn] += ek_u[i]
    ek_v_sphere[wn] += ek_v[i]
    ek_w_sphere[wn] += ek_w[i]

    # printf("i: %d\n", i)
    # printf("iter_i: %d, iter_j: %d, iter_k: %d\n", iter_i, iter_j, iter_k)

    # printf("wn: %d\n", wn)
    # printf("ek_u: %f, ek_v: %f, ek_w: %f\n", ek_u[i], ek_v[i], ek_w[i])
    # printf("ek_u_sp: %f, ek_v_sp: %f, ek_w_sp: %f\n\n", ek_u_sphere[wn], ek_v_sphere[wn], ek_w_sphere[wn])

def make_data():
    EPS = 1e-18
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
            func = compyle_1d_helper_inf_norm
        elif func_idx == 1:
            func = compyle_1d_helper_2_norm
        else:
            raise NotImplementedError
         
        func(x, y, center_x)
        y.pull()

    y_actual = y.data/2
    error = np.max(np.abs(y_actual - y_expected))
    print(f"Max error: {error}")

def make_data_multi_dim(dim, nx, order):
    es_ob = EnergySpectrum.from_example(dim=dim, nx=nx, custom_formula=None)
    es_ob.compute(order=order)
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

    eps = 0.0  # To avoid division by zero
    ek_u_sphere = np.zeros((box_radius, )) + eps
    ek_v_sphere = np.zeros((box_radius, )) + eps
    ek_w_sphere = np.zeros((box_radius, )) + eps

    ek = es_ob.ek

    return ek, ek_u, ek_v, ek_w, box_side_x, box_side_y, box_side_z, center_x, center_y, center_z, ek_u_sphere, ek_v_sphere, ek_w_sphere

def main_multi_dim(func_idx:int, dim:int, nx:int, order):
    ek, ek_u, ek_v, ek_w, box_side_x, box_side_y, box_side_z, center_x, center_y, center_z, ek_u_sphere, ek_v_sphere, ek_w_sphere =\
        make_data_multi_dim(dim=dim, nx=nx, order=order)
    
    with use_config(use_openmp=True):
        if dim == 2:
            ek_u, ek_v = wrap(ek_u.ravel(), ek_v.ravel())
            ek_u_sphere, ek_v_sphere = wrap(
                ek_u_sphere.ravel(), ek_v_sphere.ravel()
            )
            if func_idx == 0:
                func = compyle_ek_2d_helper_inf_norm
            elif func_idx == 1:
                func = compyle_ek_2d_helper_2_norm
            else:
                raise NotImplementedError
            
            func(
                ek_u, ek_v, ek_u_sphere, ek_v_sphere, box_side_y, center_x, center_y
            )        
            ek_u_sphere.pull()
            ek_v_sphere.pull()
            ek_actual = (ek_u_sphere.data + ek_v_sphere.data)/2
        elif dim == 3:
            ek_u, ek_v, ek_w = wrap(ek_u.ravel(), ek_v.ravel(), ek_w.ravel())
            ek_u_sphere, ek_v_sphere, ek_w_sphere = wrap(
                ek_u_sphere.ravel(), ek_v_sphere.ravel(), ek_w_sphere.ravel()
            )
            if func_idx == 0:
                func = compyle_ek_3d_helper_inf_norm
            elif func_idx == 1:
                func = compyle_ek_3d_helper_2_norm
            else:
                raise NotImplementedError
            func(
                ek_u, ek_v, ek_w, ek_u_sphere, ek_v_sphere, ek_w_sphere, box_side_y, box_side_z, center_x, center_y, center_z
            )
            ek_u_sphere.pull()
            ek_v_sphere.pull()
            ek_w_sphere.pull()
            ek_actual = 0.5*(
                ek_u_sphere.data + ek_v_sphere.data + ek_w_sphere.data
            )
        else:
            raise NotImplementedError

    # tmp = ek_u.data.reshape(box_side_x, box_side_y, box_side_z)
    # print(f"box_side_x: {box_side_x}, box_side_y: {box_side_y}")
    # print(f"ek_u (actual):\n{tmp}")
    # print(f"ek_u_sphere (actual):\n\t{ek_u_sphere.data}")
    # print(f"ek_v_sphere (actual):\n\t{ek_v_sphere.data}")
    # print(f"ek_w_sphere (actual):\n\t{ek_w_sphere.data}")

    # print(f"ek (actual):\n\t{ek_actual}")
    # print(f"ek (expected):\n\t{ek}")
    error = np.max(np.abs(ek_actual - ek))
    print(f"Max error:\n\t{error}")

def main(func_idx:int, dim:int, nx):
    if dim == 1:
        main_1d(func_idx)
    elif dim <= 3:
        if func_idx == 0:
            order = np.inf
        elif func_idx == 1:
            order = 2
        main_multi_dim(func_idx=func_idx, dim=dim, nx=nx, order=order)
    else:
        raise NotImplementedError
    # bord = '*'*35
    # Print run success in green color
    # print(f"\033[92m{bord} RUN SUCCESSFUL {bord}\033[0m")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    FUNC_IDX_CHOICES = [0, 1]
    DIM_CHOICES = [1, 2, 3]
    parser.add_argument(
        "-f", "--func_idx", type=int, choices=FUNC_IDX_CHOICES, default=0
    )
    parser.add_argument(
        "-d", "--dim", type=int, choices=DIM_CHOICES, default=3
    )
    parser.add_argument(
        "-n", "--nx", type=int, default=32
    )

    args = parser.parse_args()
    main(func_idx=args.func_idx, dim=args.dim, nx=args.nx)
