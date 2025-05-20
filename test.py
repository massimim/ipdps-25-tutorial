import time

from fontTools.varLib.plot import stops

import lbm
import warp as wp


# define main function
def main():
    wp.clear_kernel_cache()
    # Initialize the parameters
    params = lbm.Parameters(num_steps=3, nx=200, ny=10, prescribed_vel=0.05)
    print(params)

    # Initialize the memory
    mem = lbm.Memory(params)
    #print(mem)

    # Initialize the functions
    fun = lbm.Functions(params)
    print(fun)

    # Initialize the kernels
    kernels = lbm.Kernels(params, mem, fun)
    print(kernels)

    wp.launch(kernels.get_set_bc(),
              dim=params.grid_shape,
              inputs=[mem.bc_type],
              device="cuda")

    mem.save_bc_vtk(0)

    wp.launch(kernels.get_set_f_to_equilibrium(),
                dim=params.grid_shape,
                inputs=[mem.bc_type, mem.f_0],
                device="cuda")

    wp.launch(kernels.get_macroscopic(),
                dim=params.grid_shape,
                inputs=[mem.f_0, mem.rho, mem.u],
                device="cuda")
    mem.save_magnituge_vtk(0)


    # add timer
    start = time.time()

    for it in range(1,params.num_steps+1):
        wp.launch(kernels.get_pull_stream(),
                  dim=params.grid_shape,
                  inputs=[mem.f_0, mem.f_1],
                  device="cuda")

        wp.launch(kernels.get_apply_boundary_conditions(),
                  dim=params.grid_shape,
                  inputs=[mem.bc_type, mem.f_1],
                  device="cuda")

        if it % 1 == 0:
            wp.launch(kernels.get_macroscopic(),
                      dim=params.grid_shape,
                      inputs=[mem.f_1, mem.rho, mem.u],
                      device="cuda")
            # mem.export_warp_field_to_vti(filename=f"u_{it}.vti", u=mem.u)
            mem.save_magnituge_vtk(it)

        wp.launch(kernels.get_collision(),
                  dim=params.grid_shape,
                  inputs=[mem.f_1, params.omega],
                  device="cuda")




        # Swap the fields
        mem.f_0, mem.f_1 = mem.f_1, mem.f_0
    stop = time.time()
    enlapsed_time = stop - start
    mlups =params.compute_mlups(enlapsed_time)
    print(f"Main loop time: {enlapsed_time:5.3f} seconds")
    print(f"MLUPS:          {mlups:5.1f}")

        # Export the field to VTI

# call the main when the script is called
if __name__ == "__main__":
    main()
