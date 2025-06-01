import typing
import warp as wp
import lbm_mgpu

class Functions:
    def __init__(self, parameters: lbm_mgpu.Parameters):
        self.params = parameters

    def get_equilibrium(self):

        Macro = self.params.get_macroscopic_type()

        Q = self.params.Q
        D = self.params.D
        c_dev = self.params.c_dev
        w_dev = self.params.w_dev

        @wp.func
        def equilibrium(
                mcrpc: typing.Any,
        ):
            f_out = wp.vec(length=Q, dtype=wp.float64)

            # Compute the equilibrium
            for q in range(Q):
                # Compute cu
                cu = wp.float64(0.0)
                for d in range(D):
                    if c_dev[d, q] == 1:
                        cu += mcrpc.u[d]
                    elif c_dev[d, q] == -1:
                        cu -= mcrpc.u[d]
                cu *= wp.float64(3.0)

                # Compute usqr
                usqr = wp.float64(1.5) * wp.dot(mcrpc.u, mcrpc.u)

                # Compute feq
                f_out[q] = mcrpc.rho * w_dev[q] * (
                            wp.float64(1.0) + cu * (wp.float64(1.0) + wp.float64(0.5) * cu) - usqr)
            return f_out

        return equilibrium

    def get_macroscopic(self):
        Macro = self.params.get_macroscopic_type()

        Q = self.params.Q
        D = self.params.D
        c_dev = self.params.c_dev
        w_dev = self.params.w_dev

        @wp.func
        def zero_moment(
                f: wp.vec(length=Q, dtype=wp.float64),
        ):
            rho = wp.float64(0.0)
            for l in range(Q):
                rho += f[l]
            return rho

        @wp.func
        def first_moment(
                f: wp.vec(length=Q, dtype=wp.float64),
                rho: wp.float64,
        ):
            u_out = wp.vec(length=D, dtype=wp.float64)

            for l in range(Q):
                for d in range(D):
                    if c_dev[d, l] == 1:
                        u_out[d] += f[l]
                    elif c_dev[d, l] == -1:
                        u_out[d] -= f[l]
            u_out /= rho
            return u_out

        @wp.func
        def macroscopic(
                f: wp.vec(length=Q, dtype=wp.float64),
        ):
            mcrpc = Macro()
            # Compute the macroscopic variables
            mcrpc.rho = zero_moment(f, )
            mcrpc.u = first_moment(f, mcrpc.rho)
            return mcrpc

        return macroscopic

    def get_apply_boundary_conditions(self):
        equilibrium_fun = self.get_equilibrium()

        Macro = self.params.get_macroscopic_type()
        bc_lid = self.params.bc_lid
        bc_lid_reversed = self.params.bc_lid_reversed
        prescribed_vel = self.params.prescribed_vel
        nx = self.params.nx

        @wp.func
        def apply_boundary_conditions(
                type: wp.uint8,
        ):
            mcrpc = Macro()
            mcrpc.rho = wp.float64(1.0)
            vel = wp.float64(0.0)

            if type == bc_lid:
                vel = wp.float64(prescribed_vel)
            if type == bc_lid_reversed:
                vel = -wp.float64(prescribed_vel)


            mcrpc.u[0] = vel
            mcrpc.u[1] = wp.float64(0.0)

            f = equilibrium_fun(mcrpc)
            return f

        return apply_boundary_conditions

    def get_kbc(self):
        Macro = self.params.get_macroscopic_type()

        Q = self.params.Q
        D = self.params.D
        cc_dev = self.params.cc_dev

        # Make constants for warp
        _f_vec = wp.vec(length=Q, dtype=wp.float64)
        _pi_dim = D * (D + 1) // 2
        _pi_vec = wp.vec(_pi_dim, dtype=wp.float64)
        epsilon_host = wp.float64(1e-32)
        epsilon_dev = wp.constant(epsilon_host)

        @wp.func
        def second_moment(fneq: wp.vec(length=Q, dtype=wp.float64)):
            # Get second order moment (a symmetric tensore shaped into a vector)
            pi = _pi_vec()
            for d in range(_pi_dim):
                pi[d] = wp.float64(0.0)
                for q in range(Q):
                    pi[d] += cc_dev[q, d] * fneq[q]
            return pi

        @wp.func
        def decompose_shear_d2q9(fneq: wp.vec(length=Q, dtype=wp.float64), ):
            pi = second_moment(fneq)
            N = pi[0] - pi[2]
            s = wp.vec(length=Q, dtype=wp.float64)
            s[3] = N
            s[6] = N
            s[2] = -N
            s[1] = -N
            s[8] = pi[1]
            s[4] = -pi[1]
            s[5] = -pi[1]
            s[7] = pi[1]
            return s

        @wp.func
        def entropic_scalar_product(
                x: typing.Any,
                y: typing.Any,
                feq: typing.Any,
        ):
            e = wp.cw_div(wp.cw_mul(x, y), feq)
            e_sum = wp.float64(0.0)
            for i in range(Q):
                e_sum += e[i]
            return e_sum

        @wp.func
        def kbc(f: wp.vec(length=Q, dtype=wp.float64),
                feq: wp.vec(length=Q, dtype=wp.float64),
                mcrpc: typing.Any,
                omega: wp.float64):
            # Get second order moment (a symmetric tensore shaped into a vector)
            # Compute shear and delta_s
            fneq = f - feq

            shear = decompose_shear_d2q9(fneq)
            delta_s = shear * mcrpc.rho / wp.float64(4.0)

            # Compute required constants based on the input omega (omega is the inverse relaxation time)
            _beta = wp.float64(0.5) * wp.float64(omega)
            _inv_beta = wp.float64(1.0) / _beta

            # Perform collision
            delta_h = fneq - delta_s
            two = wp.float64(2.0)
            gamma = _inv_beta - (two - _inv_beta) * entropic_scalar_product(delta_s, delta_h, feq) / (
                    epsilon_dev + entropic_scalar_product(delta_h, delta_h, feq)
            )
            fout = f - _beta * (two * delta_s + gamma * delta_h)

            return fout

        return kbc
