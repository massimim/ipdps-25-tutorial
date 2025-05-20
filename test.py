import lbm
import warp as wp


# define main function
def main():
    # Initialize the parameters
    params = lbm.Parameters(num_steps=5000, nx=500, ny=500)
    print(params)

    # Initialize the memory
    mem = lbm.Memory(params)
    print(mem)

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

    mem.export_warp_field_to_vti(filename="bc_0.vti", u=mem.bc_type)

    wp.launch(kernels.get_set_f_to_equilibrium(),
                dim=params.grid_shape,
                inputs=[mem.bc_type, mem.f_0],
                device="cuda")

    wp.launch(kernels.get_macroscopic(),
                dim=params.grid_shape,
                inputs=[mem.f_0, mem.rho, mem.u],
                device="cuda")

    mem.export_warp_field_to_vti(filename="u_0.vti", u=mem.u)

    for it in range(1,params.num_steps+1):
        wp.launch(kernels.get_pull_stream(),
                  dim=params.grid_shape,
                  inputs=[mem.f_0, mem.f_1],
                  device="cuda")

        wp.launch(kernels.get_apply_boundary_conditions(),
                  dim=params.grid_shape,
                  inputs=[mem.bc_type, mem.f_1],
                  device="cuda")

        wp.launch(kernels.get_collision(),
                  dim=params.grid_shape,
                  inputs=[mem.f_1, params.omega],
                  device="cuda")

        if it % 1000 == 0:
            wp.launch(kernels.get_macroscopic(),
                      dim=params.grid_shape,
                      inputs=[mem.f_1, mem.rho, mem.u],
                      device="cuda")
            mem.export_warp_field_to_vti(filename=f"u_{it}.vti", u=mem.u)


        # Swap the fields
        mem.f_0, mem.f_1 = mem.f_1, mem.f_0

        # Export the field to VTI

# call the main when the script is called
if __name__ == "__main__":
    main()
