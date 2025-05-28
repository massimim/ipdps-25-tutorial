import time
from sys import prefix

from fontTools.varLib.plot import stops

import lbm
import warp as wp

exercise_name = "02_SoA_user"


# define main function
def main():
    debug = False
    wp.clear_kernel_cache()
    # Initialize the parameters
    params = lbm.Parameters(num_steps=10000,
                            nx=1024 ,
                            ny=768 ,
                            prescribed_vel=0.5,
                            Re=10000.0)
    print(params)

    f_0 = wp.zeros((params.Q,) + params.grid_shape , dtype=wp.float64)
    f_1 = wp.zeros((params.Q,) + params.grid_shape, dtype=wp.float64)

    @wp.func
    def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
        return field[card, xi, yi ]

    @wp.func
    def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                    value: wp.float64):
        field[card, xi, yi] = value

    # Initialize the memory
    mem = lbm.Memory(params,
                     f_0=f_0,
                     f_1=f_1,
                     read=read_field,
                     write=write_field)

    # Initialize the kernels
    functions = lbm.Functions(params)
    kernels = lbm.Kernels(params, mem)


    Q = params.Q
    D = params.D
    bc_bulk = params.bc_bulk
    c_dev = params.c_dev
    shape_dev = params.shape_dev

    @wp.kernel
    def stream(
            f_in: wp.array3d(dtype=wp.float64),
            f_out: wp.array3d(dtype=wp.float64),
    ):
        # Get the global index
        i, j = wp.tid()
        index = wp.vec2i(i, j)
        f_post = wp.vec(length=Q, dtype=wp.float64)

        for q in range(params.Q):
            pull_ngh = wp.vec2i(0, 0)
            outside_domain = False

            for d in range(D):
                pull_ngh[d] = index[d] - c_dev[d, q]

                if pull_ngh[d] < 0 or pull_ngh[d] >= shape_dev[d]:
                    outside_domain = True
            if not outside_domain:
                f_post[q] = read_field(field=f_in, card=q, xi=pull_ngh[0], yi=pull_ngh[1])

        # Set the output
        for q in range(params.Q):
            write_field(field=f_out, card=q, xi=index[0], yi=index[1], value=f_post[q])


    compute_boundaries = functions.get_apply_boundary_conditions()
    @wp.kernel
    def apply_boundary_conditions(
            bc_type_field: wp.array2d(dtype=wp.uint8),
            f_out: wp.array3d(dtype=wp.float64),
    ):
        # Get the global index
        i, j = wp.tid()
        index = wp.vec2i(i, j)

        bc_type = bc_type_field[index[0], index[1]]
        if bc_type == bc_bulk:
            return

        f = compute_boundaries(bc_type)

        for q in range(params.Q):
            write_field(field=f_out, card=q, xi=index[0], yi=index[1], value=f[q])

    compute_macroscopic = functions.get_macroscopic()
    compute_equilibrium = functions.get_equilibrium()
    compute_collision = functions.get_kbc()

    @wp.kernel
    def collide(
            f: wp.array3d(dtype=wp.float64),
            omega: wp.float64,
    ):
        # Get the global index
        i, j = wp.tid()
        index = wp.vec2i(i, j)
        # Get the equilibrium

        f_post_stream = wp.vec(length=Q, dtype=wp.float64)
        for q in range(params.Q):
            f_post_stream[q] = read_field(field=f, card=q, xi=index[0], yi=index[1])

        mcrpc = compute_macroscopic(f_post_stream)

        # Compute the equilibrium
        f_eq = compute_equilibrium(mcrpc)

        f_post_collision = compute_collision(f_post_stream, f_eq, mcrpc, omega)

        # Set the output
        for q in range(params.Q):
            write_field(field=f, card=q, xi=index[0], yi=index[1], value=f_post_collision[q])

    lbm.setup_LDC_problem(params=params, mem=mem)

    # #mem.save_magnituge_vtk(0)
    def iterate():
        wp.launch(stream,
                  dim=params.grid_shape,
                  inputs=[mem.f_0, mem.f_1],
                  device="cuda")

        wp.launch(apply_boundary_conditions,
                  dim=params.grid_shape,
                  inputs=[mem.bc_type, mem.f_1],
                  device="cuda")

        wp.launch(collide,
                  dim=params.grid_shape,
                  inputs=[mem.f_1, params.omega],
                  device="cuda")

        # Swap the fields
        mem.f_0, mem.f_1 = mem.f_1, mem.f_0

    # Warm up iteration
    iterate()

    # Wait for the warm-up to finish
    wp.synchronize()
    # Start timer
    start = time.time()
    for it in range(params.num_steps):
        iterate()

    wp.synchronize()
    stop = time.time()

    lbm.export_final(prefix = exercise_name, params=params, mem=mem, f=mem.f_0)


    # Statistics
    elapsed_time = stop - start
    mlups = params.compute_mlups(elapsed_time)
    print(f"Main loop time: {elapsed_time:5.3f} seconds")
    print(f"MLUPS:          {mlups:5.1f}")


# call the main when the script is called
if __name__ == "__main__":
    main()
