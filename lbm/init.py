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


def help_construct_opposite_indices(c_host):
    c = c_host.T
    return np.array([c.tolist().index((-c[i]).tolist()) for i in range(Q)])


def help_construct_lattice_moment(c_host):
    """
    This function constructs the moments of the lattice.

    The moments are the products of the velocity vectors, which are used in the computation of
    the equilibrium distribution functions and the collision operator in the Lattice Boltzmann
    Method (LBM).

    Returns
    -------
    cc: numpy.ndarray
        The moments of the lattice.
    """
    # Counter for the loop
    cntr = 0
    c = c_host.T
    # nt: number of independent elements of a symmetric tensor
    nt = D * (D + 1) // 2
    cc = np.zeros((Q, nt))
    cntr = 0
    for a in range(D):
        for b in range(a, D):
            cc[:, cntr] = c[:, a] * c[:, b]
            cntr += 1
    return cc


# ---------------------
# Grid and simulation parameters
# ---------------------
nx, ny = 128, 128
grid_shape = (nx, ny)

num_steps = 2000

# Desired Reynolds number
# Setting fluid viscosity and relaxation parameter.
Re = 200.0
prescribed_vel = 0.05
clength = grid_shape[0] - 1
visc = prescribed_vel * clength / Re
omega = 1.0 / (3.0 * visc + 0.5)

D = 2
Q = 9

cx = [0, 0, 0, 1, -1, 1, -1, 1, -1]
cy = [0, 1, -1, 0, 1, -1, 0, 1, -1]
c_host = np.array(tuple(zip(cx, cy))).T
w_host = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 36, 1 / 36])
opp_indices_host = help_construct_opposite_indices(c_host)
cc_host = help_construct_lattice_moment(c_host)
print(c_host)

sim_dtype = wp.float64
c_dev = wp.constant(wp.mat((D, Q), dtype=wp.int32)(c_host))
w_dev = wp.constant(wp.vec(Q, dtype=sim_dtype)(w_host))
opp_indices = wp.constant(wp.vec(Q, dtype=wp.int32)(opp_indices_host))
cc_dev = wp.constant(wp.mat((Q, D * (D + 1) // 2), dtype=sim_dtype)(cc_host))


# self.cc = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=dtype)(self._cc))
# self.c_float = wp.constant(wp.mat((self.d, self.q), dtype=dtype)(self._c_float))
# self.qi = wp.constant(wp.mat((self.q, self.d * (self.d + 1) // 2), dtype=dtype)(self._qi))


def help_create_field(
        cardinality: int,
        dtype,
        fill_value=None):
    shape = (cardinality,) + grid_shape
    if fill_value is None:
        f = wp.zeros(shape, dtype=dtype)
    else:
        f = wp.full(shape, fill_value, dtype=dtype)
    return f


def export_warp_field_to_vti(
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
    if field.ndim != 3:
        raise ValueError(f"u must be shape (D,nx,ny), got {field.shape}")
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


bc_bulk = wp.uint8(0)
bc_wall = wp.uint8(1)
bc_lid = wp.uint8(2)

f_0 = help_create_field(Q, sim_dtype)
f_1 = help_create_field(Q, sim_dtype)

bc_type = help_create_field(cardinality=1, dtype=wp.uint8)
u = help_create_field(cardinality=D, dtype=sim_dtype)
rho = help_create_field(cardinality=1, dtype=sim_dtype)


# Construct the equilibrium functional
class LBM_functions:
    @staticmethod
    def get_equilibrium():
        @wp.func
        def equilibrium(
                rho: wp.vec(1, dtype=sim_dtype),
                u: wp.vec(D, dtype=sim_dtype),
        ):
            f_out = wp.vec(length=Q, dtype=sim_dtype)

            # Compute the equilibrium
            for q in range(Q):
                # Compute cu
                cu = sim_dtype(0.0)
                for d in range(D):
                    if c_dev[d, q] == 1:
                        cu += u[d]
                    elif c_dev[d, q] == -1:
                        cu -= u[d]
                cu *= sim_dtype(3.0)

                # Compute usqr
                usqr = sim_dtype(1.5) * wp.dot(u, u)

                # Compute feq
                f_out[q] = rho[0] * w_dev[q] * (sim_dtype(1.0) + cu * (sim_dtype(1.0) + sim_dtype(0.5) * cu) - usqr)
            return f_out

        return equilibrium

    @staticmethod
    def get_pull_stream():
        @wp.func
        def pull_stream(
                index: typing.Any,
                f_mem: wp.array3d(dtype=sim_dtype),
        ):
            f_vec = wp.vec(length=Q, dtype=sim_dtype)
            for q in range(Q):
                pull_ngh = wp.vec3i(0, 0, 0)
                f_vec[q] = f_mem[q]

                for d in range(D):
                    pull_ngh[d] = index[d] - c_dev[d, q]

                    # impose periodicity for out of bound values
                    if pull_ngh[d] < 0:
                        pull_ngh[d] = f_mem.shape[d + 1] - 1
                    elif pull_ngh[d] >= f_mem.shape[d + 1]:
                        pull_ngh[d] = 0

                f_vec[q] = f_mem[q, pull_ngh[0], pull_ngh[1], pull_ngh[2]]

            return f_vec

        return pull_stream

    @staticmethod
    def get_macroscopic():
        @wp.func
        def zero_moment(
                f: wp.vec(length=Q, dtype=sim_dtype),
        ):
            rho = sim_dtype(0.0)
            for l in range(Q):
                rho += f[l]
            return rho

        @wp.func
        def first_moment(
                f: wp.vec(length=Q, dtype=sim_dtype),
                rho: sim_dtype,
        ):
            u_out = wp.vec(length=D, dtype=sim_dtype)

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
                f: wp.vec(length=Q, dtype=sim_dtype),
        ):
            mcrpc = wp.vec(length=4, dtype=sim_dtype)
            # Compute the macroscopic variables
            rho = zero_moment(f, )
            u = first_moment(f, rho)
            mcrpc[0] = rho
            for d in range(D):
                mcrpc[d + 1] = u[d]
            return mcrpc

        return macroscopic

    @staticmethod
    def get_kbc():
        # Make constants for warp
        _f_vec = wp.vec(length=Q, dtype=sim_dtype)
        _pi_dim = D * (D + 1) // 2
        _pi_vec = wp.vec(_pi_dim, dtype=sim_dtype)
        epsilon_host = sim_dtype(1e-32)
        epsilon_dev = wp.constant(epsilon_host)

        @wp.func
        def second_moment(fneq: wp.vec(length=Q, dtype=sim_dtype)):
            # Get second order moment (a symmetric tensore shaped into a vector)
            pi = _pi_vec()
            for d in range(_pi_dim):
                pi[d] = sim_dtype(0.0)
                for q in range(Q):
                    pi[d] += cc_dev[q, d] * fneq[q]
            return pi

        @wp.func
        def decompose_shear_d2q9(fneq: wp.vec(length=Q, dtype=sim_dtype),):
            pi = second_moment(fneq)
            N = pi[0] - pi[2]
            s = wp.vec(length=Q, dtype=sim_dtype)
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
            e_sum = sim_dtype(0.0)
            for i in range(Q):
                e_sum += e[i]
            return e_sum

        @wp.func
        def kbc(f: wp.vec(length=Q, dtype=sim_dtype),
                feq: wp.vec(length=Q, dtype=sim_dtype),
                rho: wp.vec(length=1, dtype=sim_dtype),
                u: wp.vec(length=D, dtype=sim_dtype),
                omega: sim_dtype):
            # Get second order moment (a symmetric tensore shaped into a vector)
            # Compute shear and delta_s
            fneq = f - feq

            shear = decompose_shear_d2q9(fneq)
            delta_s = shear * rho / sim_dtype(4.0)

            # Compute required constants based on the input omega (omega is the inverse relaxation time)
            _beta = sim_dtype(0.5) * sim_dtype(omega)
            _inv_beta = sim_dtype(1.0) / _beta

            # Perform collision
            delta_h = fneq - delta_s
            two = sim_dtype(2.0)
            gamma = _inv_beta - (two - _inv_beta) * entropic_scalar_product(delta_s, delta_h, feq) / (
                epsilon_dev + entropic_scalar_product(delta_h, delta_h, feq)
            )
            fout = f - _beta * (two * delta_s + gamma * delta_h)

            return fout


class LBM_kernels:
    @staticmethod
    def get_equilibrium():
        equilibrium_fun = LBM_functions.get_equilibrium()

        # Construct the warp kernel
        @wp.kernel
        def equilibrium(
                rho_in: wp.array3d(dtype=sim_dtype),
                u_in: wp.array3d(dtype=sim_dtype),
                f_out: wp.array3d(dtype=sim_dtype),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)
            # Get the equilibrium
            u = wp.vector(length=D, dtype=sim_dtype)
            rho = wp.vector(length=1, dtype=sim_dtype)

            for d in range(D):
                u[d] = u_in[d, index[0], index[1]]
            rho[0] = rho_in[0, index[0], index[1]]

            f_eq = equilibrium_fun(rho, u)

            # Set the output
            for l in range(Q):
                f_out[l, *index[0], index[1]] = f_eq[l]

        return equilibrium

    @staticmethod
    def get_set_f_to_equilibrium():
        equilibrium_fun = LBM_functions.get_equilibrium()

        @wp.kernel
        def set_f_to_equilibrium(
                bc_type: wp.array3d(dtype=wp.uint8),
                f: wp.array3d(dtype=sim_dtype),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            # Set the output
            for q in range(Q):
                f[q, index[0], index[1]] = w_dev[q]

            if bc_type[0, index[0], index[1]] == bc_lid:
                u = wp.vector(length=D, dtype=sim_dtype)
                rho = wp.vector(length=1, dtype=sim_dtype)

                rho[0] = sim_dtype(1.0)
                u[0] = sim_dtype(prescribed_vel)
                u[1] = sim_dtype(0)

                f_eq = equilibrium_fun(rho, u)
                for q in range(Q):
                    f[q, index[0], index[1]] = f_eq[q]

        return set_f_to_equilibrium

    @staticmethod
    def get_pull_stream():
        @wp.kernel
        def pull_stream(
                f_in: wp.array3d(dtype=sim_dtype),
                f_out: wp.array3d(dtype=sim_dtype),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            f_post = wp.vector(length=Q, dtype=sim_dtype)
            LBM_functions.get_pull_stream()(index, f_in, f_post)
            # Set the output
            for q in range(Q):
                f_out[q, index[0], index[1], index[2]] = f_post[q]

        return pull_stream

    @staticmethod
    def get_macroscopic():
        macroscopic_fun = LBM_functions.get_macroscopic()

        @wp.kernel
        def macroscopic(
                f_in: wp.array3d(dtype=sim_dtype),
                rho_out: wp.array3d(dtype=sim_dtype),
                u_out: wp.array3d(dtype=sim_dtype),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            f = wp.vector(length=Q, dtype=sim_dtype)

            for q in range(Q):
                f[q] = f_in[q, index[0], index[1]]

            mcrpc = macroscopic_fun(f)

            for d in range(D):
                u_out[d, index[0], index[1]] = mcrpc[d + 1]
            rho_out[0, index[0], index[1]] = mcrpc[0]

        return macroscopic

    @staticmethod
    def get_set_bc():
        @wp.kernel
        def set_bc(
                bc_type: wp.array3d(dtype=wp.uint8),
        ):
            # Get the global index
            i, j = wp.tid()
            index = wp.vec2i(i, j)

            bc_type[0, index[0], index[1]] = bc_bulk

            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                bc_type[0, index[0], index[1]] = bc_wall

            if i == nx - 1:
                bc_type[0, index[0], index[1]] = bc_lid

        return set_bc


wp.launch(LBM_kernels.get_set_bc(),
          dim=grid_shape,
          inputs=[bc_type],
          device="cuda")

# ---------------------
# Main time-stepping
# ---------------------
for step in range(1):
    wp.launch(LBM_kernels.get_set_f_to_equilibrium(),
              dim=grid_shape,
              inputs=[bc_type, f_0],
              device="cuda")
    wp.launch(LBM_kernels.get_macroscopic(),
              dim=grid_shape,
              inputs=[f_0, rho, u],
              device="cuda")
    # swap buffers
    # f, f_temp = f_temp, f

export_warp_field_to_vti(
    filename="u_0.vti",
    u=u,
)

export_warp_field_to_vti(
    filename="bc_0.vti",
    u=bc_type,
)

export_warp_field_to_vti(
    filename="f_0.vti",
    u=f_0,
)

# wp.launch(LBM_kernels.get_equilibrium(),
#           dim=grid_shape,
#           inputs=[rho, u, f_0],
#           device="cuda")
# wp.launch(LBM_kernels.get_pull_stream(),
#           dim=grid_shape,
#           inputs=[f_0, f_1],
#           device="cuda"))
