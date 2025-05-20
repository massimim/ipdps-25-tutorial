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
import lbm


class Memory:
    def __init__(self, parameters: lbm.Parameters):
        self.params = parameters

        self.f_0 = self.help_create_field(self.params.Q, self.params.sim_dtype)
        self.f_1 = self.help_create_field(self.params.Q, self.params.sim_dtype)

        self.bc_type = self.help_create_field(cardinality=1, dtype=wp.uint8)
        self.u = self.help_create_field(cardinality=self.params.D, dtype=self.params.sim_dtype)
        self.rho = self.help_create_field(cardinality=1, dtype=self.params.sim_dtype)

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

    def get_read(self):
        @wp.function
        def read_field(f, card, xi, yi):
            return f[card, xi, yi]

        return read_field

    def get_write(self):
        @wp.function
        def write_field(f, card, xi, yi, value):
            f[card, xi, yi] = value

        return write_field

    # to string method
    def __str__(self):
        return f"Memory(f_0={self.f_0}, f_1={self.f_1}, bc_type={self.bc_type}, u={self.u}, rho={self.rho})"
