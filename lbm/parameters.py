import numpy as np
import matplotlib.pyplot as plt
import warp as wp
from pyevtk.hl import imageToVTK
import typing
import numpy as np
import pyvista as pv
import warp as wp
import numpy as np
import pyvista as pv

import warp as wp
import numpy as np
import pyvista as pv


class Parameters:
    def __init__(self, nx=128, ny=128, num_steps=2000, Re=200.0, prescribed_vel=0.05):
        self.D = 2
        self.Q = 9
        self.nx = nx
        self.ny = ny
        self.grid_shape = (nx, ny)
        self.num_steps = num_steps

        # Desired Reynolds number
        # Setting fluid viscosity and relaxation parameter.
        self.Re = Re
        self.prescribed_vel = prescribed_vel
        clength = self.grid_shape[0] - 1
        visc = prescribed_vel * clength / Re
        self.omega = 1.0 / (3.0 * visc + 0.5)

        def help_construct_opposite_indices(c_host):
            c = c_host.T
            return np.array([c.tolist().index((-c[i]).tolist()) for i in range(self.Q)])

        def help_construct_lattice_moment(c_host):
            """
            This function constructs the moments of the lattice.

            The moments are the products of the velocity vectors, which are used in the computation of
            the equilibrium distribution functions and the collision operator in the Lattice Boltzmann
            Method (LBM).

            Returns
            -------
            cc: numpy.ndarray
                The moments of the lattice.
            """
            # Counter for the loop
            cntr = 0
            c = c_host.T
            # nt: number of independent elements of a symmetric tensor
            nt = self.D * (self.D + 1) // 2
            cc = np.zeros((self.Q, nt))
            cntr = 0
            for a in range(self.D):
                for b in range(a, self.D):
                    cc[:, cntr] = c[:, a] * c[:, b]
                    cntr += 1
            return cc

        cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
        cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
        self.c_host = np.array(tuple(zip(cx, cy))).T
        self.w_host = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 36, 1 / 36])
        self.opp_indices_host = help_construct_opposite_indices(self.c_host)
        self.cc_host = help_construct_lattice_moment(self.c_host)
        #print(self.c_host)

        self.sim_dtype = wp.float64
        self.c_dev = wp.constant(wp.mat((self.D, self.Q), dtype=wp.int32)(self.c_host))
        self.w_dev = wp.constant(wp.vec(self.Q, dtype=self.sim_dtype)(self.w_host))
        self.opp_indices = wp.constant(wp.vec(self.Q, dtype=wp.int32)(self.opp_indices_host))
        self.cc_dev = wp.constant(wp.mat((self.Q, self.D * (self.D + 1) // 2), dtype=self.sim_dtype)(self.cc_host))


    # add a to string method for printing the parameters
    def __str__(self):
        return f"LBM Problem Parameters(nx={self.nx}, ny={self.ny}, num_steps={self.num_steps}, Re={self.Re}, prescribed_vel={self.prescribed_vel})"
