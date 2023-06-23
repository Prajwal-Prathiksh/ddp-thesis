import numpy as np
import matplotlib.pyplot as plt
from ftle import FTLyapunovExponent
import time
import argparse

def get_ftles(lam1, lam2):
    fit = np.maximum(lam1, lam2)
    fit = np.log(np.sqrt(fit))

    bit = np.minimum(lam1, lam2)
    bit = np.log(np.sqrt(bit))
    return fit, bit

def flow_map(X, Y, ftype='spiral'):
    pi = np.pi
    sin, cos = np.sin, np.cos
    sqrt = np.sqrt
    R = X**2 + Y**2
    c_fit, c_bit = None, None
    
    if ftype == 'parabolic':
        a, b = 1.5, 1.
        x = a*X
        y = X**2 + b*Y
        
        tmp1 = 4*X**2 + a**2 - 2*a*b + b**2
        tmp2 = 4*X**2 + a**2 + 2*a*b + b**2
        lam_1 = 2*X**2 - np.sqrt(tmp1*tmp2)/2 + (a**2 + b**2)/2
        lam_2 = 2*X**2 + np.sqrt(tmp1*tmp2)/2 + (a**2 + b**2)/2

        c_fit, c_bit = get_ftles(lam_1, lam_2)
    
    elif ftype == 'exponential':
        a, b = 1., 1.
        x = a*np.exp(-X/a)
        y = b*np.exp(-Y/b)

        lam_1 = np.exp(-2*X/a)
        lam_2 = np.exp(-2*Y/b)

        c_fit, c_bit = get_ftles(lam_1, lam_2)
        
    elif ftype == 'spiral':
        a = 0.1
        x = X + a*cos(2*pi*R)
        y = Y + a*sin(2*pi*R)

        sigma_2 = 4*pi*X*a*sin(2*pi*R)
        sigma_3 = 4*pi*Y*a*cos(2*pi*R)
        sqigma_1_1 = (2*X*a*pi)**2 + (2*Y*a*pi)**2 + sigma_3 - sigma_2 + 1
        sigma_1 = 4*pi*a*sqrt(R*sqigma_1_1)
        sigma_0 = 8*(X*a*pi)**2 + 8*(Y*a*pi)**2 + sigma_3 - sigma_2 + 1

        lam_1 = sigma_0 - sigma_1
        lam_2 = sigma_0 + sigma_1

        c_fit, c_bit = get_ftles(lam_1, lam_2)
    else:
        raise ValueError('Unknown flow map type: {}'.format(ftype))
    return x, y, c_fit, c_bit

def plot_cm(
    axes, idx, x, y, s=2., c=None, title="", xlabel=r'$x$', ylabel=r'$y$',
    tick_range=[-1., 1.], tick_step=0.5
):
    axes[idx].set_title(title)
    axes[idx].scatter(x, y, s=s, c=c)
    axes[idx].set_xlabel(xlabel)
    axes[idx].set_ylabel(ylabel)

    axes[idx].set_aspect('equal', adjustable='box')
    # Set ticks
    axes[idx].set_xticks(
        np.arange(tick_range[0], tick_range[1]+tick_step, tick_step)
    )
    axes[idx].set_yticks(
        np.arange(tick_range[0], tick_range[1]+tick_step, tick_step)
    )
    
    # Draw horizontal and vertical lines at x = -1, 1, y = -1, 1
    axes[idx].axhline(y=-1, color='k', linestyle='--')
    axes[idx].axhline(y=1, color='k', linestyle='--')
    axes[idx].axvline(x=-1, color='k', linestyle='--')
    axes[idx].axvline(x=1, color='k', linestyle='--')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--flow-type', type=str, default='exp',
        help='Flow type.', choices=['parabolic', 'spiral', 'exponential'],
        dest='flow_type'
    )
    args = parser.parse_args()
    flow_type = args.flow_type

    nx = 50
    _x = np.arange(-1, 1+1./nx, 1./nx)
    X, Y = np.meshgrid(_x, _x)

    x, y, c_fit, c_bit = flow_map(X=X, Y=Y, ftype=flow_type)
    ftle_ob = FTLyapunovExponent.from_example(dim=2, nx=40, flow_type=flow_type)

    # FIT
    t0 = time.time()
    ftle_fit_res = ftle_ob.compute(ftle_type='forward', mode='mpi')
    t1 = time.time()
    print(f"Computed forward FTLE in {t1-t0} seconds")

    # BIT
    t0 = time.time()
    ftle_bit_res = ftle_ob.compute(ftle_type='backward', mode="mpi")
    t1 = time.time()
    print(f"Computed backward FTLE in {t1-t0} seconds")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()
    plot_cm(
        axes=axes, idx=0, x=X, y=Y, s=2., c=c_fit,
        xlabel=r'$X$', ylabel=r'$Y$',
        title=r"$B_0$ - FTLE (FIT) Exact",
    )

    X_comp = ftle_ob.pa_0.get('x')
    Y_comp = ftle_ob.pa_0.get('y')
    c_fit_comp = ftle_fit_res
    plot_cm(
        axes=axes, idx=1, x=X_comp, y=Y_comp, s=2, c=c_fit_comp,
        xlabel=r'$X$', ylabel=r'$Y$',
        title=r"$B_0$ - FTLE (FIT) Computed",
    )

    plot_cm(
        axes=axes, idx=2, x=x, y=y, s=2., c=c_bit,
        xlabel=r'$X$', ylabel=r'$Y$',
        title=r"$B_t$ - FTLE (BIT) Exact",
    )

    x_comp = ftle_ob.pa_f.get('x')
    y_comp = ftle_ob.pa_f.get('y')
    c_bit_comp = ftle_bit_res
    plot_cm(
        axes=axes, idx=3, x=x_comp, y=y_comp, s=2, c=c_bit_comp,
        xlabel=r'$X$', ylabel=r'$Y$',
        title=r"$B_t$ - FTLE (BIT) Computed",
    )

    # Add one colorbar for all plots
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(
        axes[0].collections[0], cax=cbar_ax,
        label=r'$\lambda$'
    )

    fname = f"ftle_{flow_type}.png"
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    