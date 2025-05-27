import time

from fontTools.varLib.plot import stops

import lbm
import warp as wp
exercise = "01_AoS"

# define main function
def main():
    debug = False
    wp.clear_kernel_cache()
    # Initialize the parameters
    params = lbm.Parameters(num_steps=1000,
                            nx=1024 ,
                            ny=768 ,
                            prescribed_vel=0.5,
                            Re=10000.0)
    print(params)

    f_0 = wp.zeros(params.grid_shape + (params.Q,), dtype=wp.float64)
    f_1 = wp.zeros(params.grid_shape + (params.Q,), dtype=wp.float64)

    @wp.func
    def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
        return field[xi, yi, card]

    @wp.func
    def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                    value: wp.float64):
        field[xi, yi, card] = value

    # Initialize the memory
    mem = lbm.Memory(params,
                     f_0=f_0,
                     f_1=f_1,
                     read=read_field,
                     write=write_field)

    # Initialize the kernels
    kernels = lbm.Kernels(params, mem)

    wp.launch(kernels.get_set_lid_problem(),
              dim=params.grid_shape,
              inputs=[mem.bc_type],
              device="cuda")
    wp.launch(kernels.get_set_f_to_equilibrium(),
              dim=params.grid_shape,
              inputs=[mem.bc_type, mem.f_0],
              device="cuda")


    # #mem.save_magnituge_vtk(0)
    def iterate():
        wp.launch(kernels.get_pull_stream(),
                  dim=params.grid_shape,
                  inputs=[mem.f_0, mem.f_1],
                  device="cuda")

        wp.launch(kernels.get_apply_boundary_conditions(),
                  dim=params.grid_shape,
                  inputs=[mem.bc_type, mem.f_1],
                  device="cuda")

        wp.launch(kernels.get_collision(kbc=True),
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

    for it in range(params.num_steps ):
        iterate()

    wp.synchronize()
    stop = time.time()

    # Compute the macroscopic variables
    wp.launch(kernels.get_macroscopic(),
              dim=params.grid_shape,
              inputs=[mem.f_1, mem.rho, mem.u],
              device="cuda")
    mem.export_final(exercise)

    # Statistics
    elapsed_time = stop - start
    mlups = params.compute_mlups(elapsed_time)
    print(f"Main loop time: {elapsed_time:5.3f} seconds")
    print(f"MLUPS:          {mlups:5.1f}")

    # Export the field to VTI


# call the main when the script is called
if __name__ == "__main__":
    main()
