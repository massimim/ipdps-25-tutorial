import lbm
import wp

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

# call the main when the script is called
if __name__ == "__main__":
    main()