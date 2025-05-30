import warp as wp
import lbm


class Kernels:
    def __init__(self, parameters: lbm.Parameters,
                 memory: lbm.Memory,
                 user_pull_stream = None ):
        self.params = parameters
        self.memory = memory
        self.functions = lbm.Functions(parameters)
        self.stream = lbm.stream.Stream(parameters, memory.get_read())
        self.user_pull_stream = user_pull_stream

    def get_set_f_to_equilibrium(self):
        equilibrium_fun = self.functions.get_equilibrium()

        Macro = self.params.get_macroscopic_type()
        
        Q = self.params.Q
        D = self.params.D

        w_dev = self.params.w_dev
        bc_lid = self.params.bc_lid
        prescribed_vel = self.params.prescribed_vel

        write = self.memory.get_write()
        @wp.kernel
        def set_f_to_equilibrium(
                bc_type: wp.array2d(dtype=wp.uint8),
                f: wp.array3d(dtype=wp.float64),
        ):
            # Get the global index
            ix, iy = wp.tid()

            # Set the output
            for q in range(Q):
                write(field=f, card=q, xi=ix, yi=iy, value=w_dev[q])

            if bc_type[ix, iy] == bc_lid:
                mcrpc = Macro()
                mcrpc.rho = wp.float64(1.0)
                mcrpc.u[0] = wp.float64(prescribed_vel)
                mcrpc.u[1] = wp.float64(0)

                f_eq = equilibrium_fun(mcrpc)
                for q in range(Q):
                    write(field=f, card=q, xi=ix, yi=iy, value=f_eq[q])

        return set_f_to_equilibrium

    def get_macroscopic(self):
        macroscopic_fun = self.functions.get_macroscopic()

        Q = self.params.Q
        D = self.params.D
        read = self.memory.get_read()

        @wp.kernel
        def macroscopic(
                f_in: wp.array3d(dtype=wp.float64),
                rho_out: wp.array2d(dtype=wp.float64),
                u_out: wp.array3d(dtype=wp.float64),
        ):
            # Get the global index
            ix, iy = wp.tid()

            f = wp.vector(length=Q, dtype=wp.float64)

            for q in range(Q):
                f[q] = read(field = f_in, card = q, xi=ix, yi=iy)

            mcrpc = macroscopic_fun(f)

            for d in range(D):
                u_out[d, ix,iy] = mcrpc.u[d]

            rho_out[ix, iy] = mcrpc.rho

        return macroscopic

    def get_set_lid_problem(self):

        bc_lid = self.params.bc_lid
        bc_wall = self.params.bc_wall
        bc_bulk = self.params.bc_bulk
        nx, ny = self.params.dim

        @wp.kernel
        def set_bc(
                bc_type: wp.array2d(dtype=wp.uint8),
        ):
            # Get the global index
            ix, iy = wp.tid()
            #wp.printf("Setting lid at %d, %d\n", ix, iy)

            bc_type[ix, iy ] = bc_bulk

            if ix == 0 or iy == 0 or ix == nx - 1:
                bc_type[ix, iy] = bc_wall
                return

            if iy == ny - 1 and (ix != 0 and ix != nx - 1):
                bc_type[ix, iy] = bc_lid

        return set_bc

    def get_set_02_problem(self, length:wp.int32):

        bc_lid = self.params.bc_lid
        bc_lid_reversed = self.params.bc_lid_reversed
        bc_wall = self.params.bc_wall
        bc_bulk = self.params.bc_bulk
        nx, ny = self.params.grid_shape
        half_len = length // 2
        @wp.kernel
        def set_bc(
                bc_type: wp.array2d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            bc_type[index[0], index[1]] = bc_bulk

            if i == 0 or i == 0 or i == nx - 1:
                bc_type[index[0], index[1]] = bc_wall
                return

            o = wp.vec2f(wp.float32(nx // 2),wp.float32( ny // 2))
            p = wp.vec2f(wp.float32(i),wp.float32( j))

            d = wp.norm_l2(p-o)
            if d < wp.float32(half_len):
                bc_type[index[0], index[1]] = bc_wall
                return

            if (j == ny - 1 or j==0)  and (i != 0 and i != nx - 1):
                bc_type[index[0], index[1]] = bc_lid
                if i > nx // 2:
                    bc_type[index[0], index[1]] = bc_lid_reversed

        return set_bc