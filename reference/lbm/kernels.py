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

    def get_equilibrium(self):
        equilibrium_fun = self.functions.get_equilibrium()
        
        Q = self.params.Q
        D = self.params.D

        read = self.memory.get_read()
        write = self.memory.get_write()

        # Construct the warp kernel
        @wp.kernel
        def equilibrium(
                rho_in: wp.array2d(dtype=wp.float64),
                u_in: wp.array3d(dtype=wp.float64),
                f_out: wp.array3d(dtype=wp.float64),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)
            # Get the equilibrium
            u = wp.vector(length=D, dtype=wp.float64)
            rho = wp.vector(length=wp.int32(1), dtype=wp.float64)

            for d in range(D):
                u[d] = u_in[d, index[0], index[1]]
            rho[0] = rho_in[index[0], index[1]]

            f_eq = equilibrium_fun(rho, u)

            # Set the output
            for l in range(Q):
                write(field=f_out,
                      card=l,
                      xi=[0],
                      yi=index[1],
                      value=f_eq[l])

        return equilibrium

    def get_collision(self, kbc=False):
        equilibrium_fun = self.functions.get_equilibrium()
        macroscopic_fun = self.functions.get_macroscopic()
        collision_fun = self.functions.get_kbc()
        if not kbc:
            collision_fun = self.functions.get_bgk()

        equilibrium_fun = self.functions.get_equilibrium()
        
        Q = self.params.Q
        D = self.params.D

        read = self.memory.get_read()
        write = self.memory.get_write()

        # Construct the warp kernel
        @wp.kernel
        def collision(
                f: wp.array3d(dtype=wp.float64),
                omega: wp.float64,
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)
            # Get the equilibrium
            u = wp.vector(length=D, dtype=wp.float64)
            rho = wp.vector(length=1, dtype=wp.float64)
            f_post_stream = wp.vector(length=Q, dtype=wp.float64)

            for q in range(Q):
                f_post_stream[q] = read(field=f, card=q, xi=index[0], yi=index[1])

            mcrpc = macroscopic_fun(f_post_stream)

            # Compute the equilibrium
            f_eq = equilibrium_fun(mcrpc)

            f_post_collision = collision_fun(f_post_stream, f_eq, mcrpc, omega)

            # Set the output
            for q in range(Q):
                write(field=f, card=q, xi=index[0], yi=index[1], value=f_post_collision[q])

        return collision


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
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Set the output
            for q in range(Q):
                write(field=f, card=q, xi=index[0], yi=index[1], value=w_dev[q])

            if bc_type[index[0], index[1]] == bc_lid:
                mcrpc = Macro()
                mcrpc.rho = wp.float64(1.0)
                mcrpc.u[0] = wp.float64(prescribed_vel)
                mcrpc.u[1] = wp.float64(0)

                f_eq = equilibrium_fun(mcrpc)
                for q in range(Q):
                    write(field=f, card=q, xi=index[0], yi=index[1], value=f_eq[q])

        return set_f_to_equilibrium

    def get_pull_stream(self):
        pull_stream_fun = None
        if self.user_pull_stream is not None:
            pull_stream_fun = self.user_pull_stream
        else:
            pull_stream_fun = self.stream.get_pull_stream()

        Macro = self.params.get_macroscopic_type()
        
        Q = self.params.Q
        D = self.params.D

        w_dev = self.params.w_dev
        bc_lid = self.params.bc_lid
        prescribed_vel = self.params.prescribed_vel

        write = self.memory.get_write()

        @wp.kernel
        def pull_stream(
                f_in: wp.array3d(dtype=wp.float64),
                f_out: wp.array3d(dtype=wp.float64),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)
            f_post = pull_stream_fun(index, f_in)
            # Set the output
            for q in range(Q):
                write(field=f_out, card=q, xi=index[0], yi=index[1], value=f_post[q])

        return pull_stream

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
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            f = wp.vector(length=Q, dtype=wp.float64)

            for q in range(Q):
                f[q] = read(field = f_in, card = q, xi=index[0], yi=index[1])

            mcrpc = macroscopic_fun(f)

            for d in range(D):
                u_out[d, index[0], index[1]] = mcrpc.u[d]

            rho_out[index[0], index[1]] = mcrpc.rho

        return macroscopic

    def get_apply_boundary_conditions(self):
        apply_boundary_conditions_fun = self.functions.get_apply_boundary_conditions()
        
        Q = self.params.Q

        bc_bulk = self.params.bc_bulk

        read = self.memory.get_read()
        write = self.memory.get_write()

        @wp.kernel
        def apply_boundary_conditions(
                bc_type: wp.array2d(dtype=wp.uint8),
                f_out: wp.array3d(dtype=wp.float64),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            type = bc_type[index[0], index[1]]
            if type == bc_bulk:
                return

            f = apply_boundary_conditions_fun(type)

            for q in range(Q):
                write(field=f_out, card=q, xi=index[0], yi=index[1], value=f[q])

        return apply_boundary_conditions

    def get_set_lid_problem(self):

        bc_lid = self.params.bc_lid
        bc_wall = self.params.bc_wall
        bc_bulk = self.params.bc_bulk
        nx, ny = self.params.grid_shape

        @wp.kernel
        def set_bc(
                bc_type: wp.array2d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            bc_type[index[0], index[1]] = bc_bulk

            if i == 0 or j == 0 or i == nx - 1:
                bc_type[index[0], index[1]] = bc_wall
                return

            if j == ny - 1 and (i != 0 and i != nx - 1):
                bc_type[index[0], index[1]] = bc_lid

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