import matplotlib.pyplot as plt
from matplotlib import cm

import warp as wp
import numpy as np
#import pyvista as pv
import lbm
import os
import time


class Memory:
    def help_create_field(self,
                          cardinality: int,
                          dtype,
                          fill_value=None):
        if cardinality == 1:
            shape = self.params.grid_shape
        else:
            shape = (cardinality,) + self.params.grid_shape
        if fill_value is None:
            f = wp.zeros(shape, dtype=dtype)
        else:
            f = wp.full(shape, fill_value, dtype=dtype)
        return f

    def __init__(self, parameters: lbm.Parameters):
        self.params = parameters
        self.export = lbm.Export(self.params)

        self.f_0 = self.help_create_field(self.params.Q, wp.float64)
        self.f_1 = self.help_create_field(self.params.Q, wp.float64)

        self.bc_type = self.help_create_field(cardinality=1, dtype=wp.uint8)
        self.u = self.help_create_field(cardinality=self.params.D, dtype=wp.float64)
        self.rho = self.help_create_field(cardinality=1, dtype=wp.float64)

    def get_read(self):
        """
        Get a warp function to write into an LBM population field.
        """
        @wp.func
        def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
            return field[card, xi, yi]

        return read_field

    def get_write(self):
        """
        Get a warp function to write into an LBM population field.
        """
        @wp.func
        def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                        value: wp.float64):
            field[card, xi, yi] = value

        return write_field

    def save_magnituge_vtk(self, timestep, prefix):
        u = self.u.numpy()
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        fields = {"u_magnitude": u_magnitude,
                  "u_x": u[0],
                  "u_y": u[1]}
        self.export.save_fields_vtk(fields, timestep, prefix=prefix)

    def save_magnituge_img(self, timestep, prefix):
        u = self.u.numpy()
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        self.export.save_image(u_magnitude, timestep, prefix=prefix)

    def save_bc_vtk(self, timestep):
        bc_type = self.bc_type.numpy()
        fields = {"bc_type": bc_type}
        self.export.save_fields_vtk(fields, timestep)

    # to string method
    def __str__(self):
        return f"Memory(f_0={self.f_0}, f_1={self.f_1}, bc_type={self.bc_type}, u={self.u}, rho={self.rho})"

    def image(self, it):
        if it % self.params.export_frequency == 0:
            # if self.params.export_vtk:
            #     self.save_magnituge_vtk(it, prefix=self.params.export_prefix+"_u")
            #     return
            if self.params.export_img:
                self.save_magnituge_img(it, prefix=self.params.export_prefix+"_u")
                return

    def export_final(self):
        self.save_magnituge_img(self.params.num_steps, prefix=self.params.export_prefix + "_u")
