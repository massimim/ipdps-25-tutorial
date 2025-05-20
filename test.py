import lbm
import warp as wp


# define main function
def main():
    # Initialize the parameters
    params = lbm.Parameters()
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

# call the main when the script is called
if __name__ == "__main__":
    main()
