import matplotlib.pyplot as plt
from matplotlib import cm

import warp as wp
import numpy as np
#import pyvista as pv
import lbm
import os
import time


class Export:
    def __init__(self, parameters: lbm.Parameters):
        self.params = parameters

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
