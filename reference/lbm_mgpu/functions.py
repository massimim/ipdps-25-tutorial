import typing
import warp as wp
import lbm


class Functions:
    def __init__(self, parameters: lbm.Parameters):
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
