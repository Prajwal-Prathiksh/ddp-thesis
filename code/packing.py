# creating packed configuration for non linear boundary

from pysph.base.utils import get_particle_array
from pysph.tools.geometry import remove_overlap_particles
from pysph.tools.packer import Packer, get_packing_folders

from config_mms import boundary_curve, boundary_normal
import numpy as np


def get_marrone_boundary(L, dx):
    nl = 8 * dx
    domain = 3
    x = np.mgrid[0-nl+dx/2:L+nl:dx]
    y = boundary_curve(x, L, domain)

    xn, yn = boundary_normal(x, domain)

    xs = []
    ys = []
    xns = []
    yns = []
    for i in range(8):
        shift = (i+0.5)*dx
        _x = x - shift * xn
        _y = y - shift * yn
        xs.extend(list(_x))
        ys.extend(list(_y))
        xns.extend(list(xn))
        yns.extend(list(yn))

    return np.array(xs), np.array(ys), np.array(xns), np.array(yns),


class Packing(Packer):
    def create_particles(self):
        s = self.scheme
        L = 1.0
        dx = self.dx
        nl = 8 * dx
        xf, yf = np.mgrid[dx / 2:L:dx, dx / 2:1.5 * L:dx]
        xf, yf = [np.ravel(t) for t in (xf, yf)]


        # Initialize
        volume = dx**2
        m = volume
        h = 1.2 * dx
        rho = 1.0

        free = get_particle_array(name='free', x=xf, y=yf, m=m, rho=1.0, h=h)

        xn = np.mgrid[0:1+dx/4:dx/2]
        yn = boundary_curve(xn, L, domain=3)
        nodes = s.create_boundary_node(
                    None, [xn, yn], shift=True, name='nodes')

        xs, ys = np.mgrid[dx / 2 - nl:L + nl:dx, dx / 2 - nl:1.5*L + nl:dx]
        cond0 = ~((xs>0) & (xs<L) & (ys>0) & (ys<1.5*L))
        frozen = get_particle_array(
            name='frozen', x=xs[cond0], y=ys[cond0], m=m, h=h, rho=1.0)

        boundary = get_particle_array(name='boundary')

        remove_overlap_particles(frozen, free, self.dx, dim=self.dim)

        particles = [free, frozen, nodes, boundary]

        s.setup_properties(particles)
        for pa in particles:
            pa.dt_adapt[:] = 1e20
        return particles

    def post_process(self, info_fname):
        import os
        from pysph.solver.utils import load
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return
        res = self.out
        filename = self.output_files[-1]
        data = load(filename)
        free = data['arrays']['free']
        boundary = data['arrays']['boundary']

        xf = free.x
        yf = free.y
        xb = boundary.x
        yb = boundary.y
        xf = np.concatenate((xf, xb))
        yf = np.concatenate((yf, yb))

        np.savez(self.out, xf=xf, yf=yf)


if __name__ == '__main__':
    resolutions = [50, 100, 200, 250, 400, 500, 1000]
    for nx in resolutions:
        dx = 1.0/nx
        pre, layer, res = get_packing_folders('./', dx=dx)
        print(res)
        app = Packing(None, pre, None, None, dx, res)
        app.run()
        app.post_process(app.info_filename)