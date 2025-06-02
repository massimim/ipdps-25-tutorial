import warp as wp
import numpy as np
import lbm_mgpu


class Memory:
    def help_create_field(self,
                          partitions,
                          cardinality: int,
                          dtype,
                          fill_value=None):
        fields = []
        for i, partition in enumerate(partitions):
            if cardinality == 1:
                shape = partition.shape
            elif cardinality == 2:
                nx = partition.shape[0]
                ny = partition.shape[1]
                shape = (cardinality, nx, ny)
            else:
                nx = partition.shape_with_halo[0]
                ny = partition.shape_with_halo[1]
                shape = (cardinality, nx, ny)
            if fill_value is None:
                f = wp.zeros(shape, dtype=dtype, device=self.params.gpus[i])
            else:
                f = wp.full(shape, fill_value, dtype=dtype,device=self.params.gpus[i])
            fields.append(f)
        return fields

    def __init__(self,
                 parameters: lbm_mgpu.Parameters,
                 partitions,
                 f_0=None,
                 f_1=None,
                 read=None,
                 write=None
                 ):
        """
        Initialize the memory for the Lattice Boltzmann Method (LBM) simulation.
        :param parameters:
        :param f_0:
        :param f_1:
        :param read:
        :param write:
        """
        self.params = parameters
        self.export = lbm_mgpu.Export(self.params)
        self.partitions = partitions

        if f_0 is None:
            self.f_0 = self.help_create_field(self.params.Q, wp.float64)
        else:
            self.f_0 = f_0

        if f_1 is None:
            self.f_1 = self.help_create_field(self.params.Q, wp.float64)
        else:
            self.f_1 = f_1

        self.bc_type = self.help_create_field(self.partitions, cardinality=1, dtype=wp.uint8)
        self.u = self.help_create_field(self.partitions, cardinality=self.params.D, dtype=wp.float64)
        self.rho = self.help_create_field(self.partitions, cardinality=1, dtype=wp.float64)

        self.read_fun = read
        self.write_fun = write

    def get_read(self):
        """
        Get a warp function to write into an LBM population field.
        """
        if self.read_fun is not None:
            return self.read_fun

        @wp.func
        def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
            return field[card, xi, yi + 1]

        return read_field

    def get_write(self):
        """
        Get a warp function to write into an LBM population field.
        """
        if self.write_fun is not None:
            return self.write_fun

        @wp.func
        def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                        value: wp.float64):
            field[card, xi, yi + 1] = value

        return write_field

    def save_magnituge_vtk(self, timestep, prefix):
        u = self.u.numpy()
        u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
        fields = {"u_magnitude": u_magnitude,
                  "u_x": u[0],
                  "u_y": u[1]}
        self.export.save_fields_vtk(fields, timestep, prefix=prefix)

    def save_magnituge_img(self, timestep, prefix):
        u_magnitude_list = []
        for i, p in enumerate(self.partitions):
            u = self.u[i].numpy()
            u_magnitude = (u[0] ** 2 + u[1] ** 2) ** 0.5
            u_magnitude_list.append(u_magnitude)
        u_magnitude = np.concatenate(u_magnitude_list, axis=0)
        self.export.save_image(u_magnitude, timestep, prefix=prefix)

    def save_bc_img(self, timestep, prefix):
        bctype_list = []
        for i, p in enumerate(self.partitions):
            bctype = self.bc_type[i].numpy()
            bctype_list.append(bctype)
        bctype = np.concatenate(bctype_list, axis=0)
        self.export.save_image(bctype, timestep, prefix=prefix)

    def save_bc_vtk(self, timestep):
        bc_type = self.bc_type.numpy()
        fields = {"bc_type": bc_type}
        self.export.save_fields_vtk(fields, timestep)

    # to string method
    def __str__(self):
        return f"Memory(f_0={self.f_0}, f_1={self.f_1}, bc_type={self.bc_type}, u={self.u}, rho={self.rho})"

    def image(self, it):
        wp.synchronize()
        if it % self.params.export_frequency == 0:
            if self.params.export_img:
                self.save_magnituge_img(it, prefix=self.params.export_prefix + "_u")
                return

    def image_forced(self, it):
        wp.synchronize()
        self.save_magnituge_img(it, prefix=self.params.export_prefix + "_u")
        return

    def export_final(self, example_name=None):
        wp.synchronize()

        if example_name is None:
            example_name = self.params.export_prefix

        self.save_magnituge_img(self.params.num_steps, prefix=example_name + "_u")

    def export_problem_setup(self, example_name=None):
        wp.synchronize()

        if example_name is None:
            example_name = self.params.export_prefix

        self.save_bc_img(0, prefix=example_name + "_bc")