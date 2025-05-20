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
import os
import time


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
        sim_dtype = self.params.sim_dtype

        @wp.func
        def read_field(field: wp.array3d(dtype=sim_dtype), card: wp.int32, xi: wp.int32, yi: wp.int32):
            return field[card, xi, yi]

        return read_field

    def get_write(self):
        sim_dtype = self.params.sim_dtype

        @wp.func
        def write_field(field: wp.array3d(dtype=sim_dtype), card: wp.int32, xi: wp.int32, yi: wp.int32,
                        value: sim_dtype):
            field[card, xi, yi] = value

        return write_field

    def save_magnituge_vtk(self, timestep):
        u = self.u.numpy()
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        fields = {"u_magnitude": u_magnitude,
                  "u_x": u[0],
                  "u_y": u[1]}
        self.save_fields_vtk(fields, timestep, prefix="u")

    def save_bc_vtk(self, timestep):
        bc_type = self.bc_type.numpy()
        fields = {"bc_type": bc_type}
        self.save_fields_vtk(fields, timestep)

    def save_fields_vtk(self, fields, timestep, output_dir=".", prefix="fields"):
        """
        Save VTK fields to the specified directory.

        Parameters
        ----------
        timestep (int): The timestep number to be associated with the saved fields.
        fields (Dict[str, np.ndarray]): A dictionary of fields to be saved. Each field must be an array-like object
            with dimensions (nx, ny) for 2D fields or (nx, ny, nz) for 3D fields, where:
                - nx : int, number of grid points along the x-axis
                - ny : int, number of grid points along the y-axis
                - nz : int, number of grid points along the z-axis (for 3D fields only)
            The key value for each field in the dictionary must be a string containing the name of the field.
        output_dir (str, optional, default: '.'): The directory in which to save the VTK files. Defaults to the current directory.
        prefix (str, optional, default: 'fields'): A prefix to be added to the filename. Defaults to 'fields'.

        Returns
        -------
        None

        Notes
        -----
        This function saves the VTK fields in the specified directory, with filenames based on the provided timestep number
        and the filename. For example, if the timestep number is 10 and the file name is fields, the VTK file
        will be saved as 'fields_0000010.vtk'in the specified directory.

        """
        # Assert that all fields have the same dimensions
        for key, value in fields.items():
            if key == list(fields.keys())[0]:
                dimensions = value.shape
            else:
                assert value.shape == dimensions, "All fields must have the same dimensions!"

        output_filename = os.path.join(output_dir, prefix + "_" + f"{timestep:07d}.vtk")

        # Add 1 to the dimensions tuple as we store cell values
        dimensions = tuple([dim + 1 for dim in dimensions])

        # Create a uniform grid
        if value.ndim == 2:
            dimensions = dimensions + (1,)

        grid = pv.ImageData(dimensions=dimensions)

        # Add the fields to the grid
        for key, value in fields.items():
            grid[key] = value.flatten(order="F")

        # Save the grid to a VTK file
        start = time.time()
        grid.save(output_filename, binary=True)
        print(f"Saved {output_filename} in {time.time() - start:.6f} seconds.")

    def export_warp_field_to_vti(self,
                                 filename: str,
                                 u: wp.array,
                                 origin: tuple = (0.0, 0.0, 0.0),
                                 spacing: tuple = (1.0, 1.0, 1.0),
                                 name: str = "field",
                                 pad_to_3: bool = True
                                 ):
        """
        Export a Warp array u of shape (D, nx, ny) to .vti for ParaView.

        D=1 → scalar
        D=2 → 2-comp vector (padded to 3 if pad_to_3=True)
        D=3 → 3-comp vector
        D=9 → 9-comp generic array (no VTK-level tensor tag)
        """
        # bring back to NumPy
        field = u.numpy()  # shape (D, nx, ny)

        if field.ndim == 2:
            field = field[np.newaxis, ...]
        elif field.ndim != 3:
            raise ValueError(f"u must be shape (nx,ny) or (D,nx,ny), got {field.shape}")
        D, nx, ny = field.shape
        if D not in (1, 2, 3, 9):
            raise ValueError("D must be 1, 2, 3 or 9")

        # build a flat slab
        grid = pv.ImageData(
            dimensions=(nx, ny, 1),
            spacing=spacing,
            origin=origin
        )

        # flatten components
        flats = [field[i].ravel(order="C") for i in range(D)]

        if D == 1:
            grid.point_data[name] = flats[0]

        elif D == 2:
            fx, fy = flats
            if pad_to_3:
                fz = np.zeros_like(fx)
                arr = np.column_stack((fx, fy, fz))
            else:
                arr = np.column_stack((fx, fy))
            grid.point_data[name] = arr

        elif D == 3:
            fx, fy, fz = flats
            arr = np.column_stack((fx, fy, fz))
            grid.point_data[name] = arr

        else:  # D == 9
            tensor9 = np.stack(flats, axis=1)  # shape (nx*ny, 9)
            # attach as a generic 9-component array
            grid.point_data[name] = tensor9
            # DO NOT call SetActiveTensors here — ParaView will load it stably.

        # write out
        grid.save(filename)
        kind = {1: "scalar", 2: "2-comp vector", 3: "3-comp vector", 9: "9-comp array"}[D]
        if D == 2 and pad_to_3:
            kind += " (padded to 3)"
        print(f"Wrote {kind} '{name}' → {filename}")

    # to string method
    def __str__(self):
        return f"Memory(f_0={self.f_0}, f_1={self.f_1}, bc_type={self.bc_type}, u={self.u}, rho={self.rho})"
