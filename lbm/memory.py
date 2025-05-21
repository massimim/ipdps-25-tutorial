import matplotlib.pyplot as plt
from matplotlib import cm

import warp as wp
import numpy as np
import pyvista as pv
import lbm
import os
import time


class Memory:
    def __init__(self, parameters: lbm.Parameters):
        self.params = parameters

        self.f_0 = self.help_create_field(self.params.Q, wp.float64)
        self.f_1 = self.help_create_field(self.params.Q, wp.float64)

        self.bc_type = self.help_create_field(cardinality=1, dtype=wp.uint8)
        self.u = self.help_create_field(cardinality=self.params.D, dtype=wp.float64)
        self.rho = self.help_create_field(cardinality=1, dtype=wp.float64)

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
        

        @wp.func
        def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
            return field[card, xi, yi]

        return read_field

    def get_write(self):
        

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
        self.save_fields_vtk(fields, timestep, prefix=prefix)

    def save_magnituge_img(self, timestep, prefix):
        u = self.u.numpy()
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        self.save_image(u_magnitude, timestep, prefix=prefix)

    def save_bc_vtk(self, timestep):
        bc_type = self.bc_type.numpy()
        fields = {"bc_type": bc_type}
        self.save_fields_vtk(fields, timestep)

    def save_image(self, fld, timestep=None, prefix=None, **kwargs):
        """
        Save an image of a field at a given timestep.

        Parameters
        ----------
        timestep : int
            The timestep at which the field is being saved.
        fld : jax.numpy.ndarray
            The field to be saved. This should be a 2D or 3D JAX array. If the field is 3D, the magnitude of the field will be calculated and saved.
        prefix : str, optional
            A prefix to be added to the filename. The filename will be the name of the main script file by default.

        Returns
        -------
        None

        Notes
        -----
        This function saves the field as an image in the PNG format.
        The filename is based on the name of the main script file, the provided prefix, and the timestep number.
        If the field is 3D, the magnitude of the field is calculated and saved.
        The image is saved with the 'nipy_spectral' colormap and the origin set to 'lower'.
        """

        fname = prefix

        if timestep is not None:
            fname = fname + "_" + str(timestep).zfill(4)

        if len(fld.shape) > 3:
            raise ValueError("The input field should be 2D!")
        if len(fld.shape) == 3:
            fld = np.sqrt(fld[0, ...] ** 2 + fld[0, ...] ** 2)

        plt.clf()
        kwargs.pop("cmap", None)
        plt.imsave(fname + ".png", fld.T, cmap=cm.nipy_spectral, origin="lower", **kwargs)

    def save_fields_vtk(self, fields, timestep, output_dir=".", prefix="fields"):
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

    def export(self, it):
        if it % self.params.export_frequency == 0:
            if self.params.export_vtk:
                self.save_magnituge_vtk(it, prefix=self.params.export_prefix+"_u")
                return
            if self.params.export_img:
                self.save_magnituge_img(it, prefix=self.params.export_prefix+"_u")
                return

    def export_final(self):
        self.save_magnituge_img(self.params.num_steps, prefix=self.params.export_prefix + "_u")
