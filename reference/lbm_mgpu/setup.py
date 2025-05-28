import typing
import warp as wp
import lbm

def setup_LDC_problem(params, partitions, mem):
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
    wp.synchronize()

def export_final(prefix, params, mem, f):
    # Compute the macroscopic variables
    wp.synchronize()
    kernels = lbm.Kernels(params, mem)
    wp.launch(kernels.get_macroscopic(),
              dim=mem.params.grid_shape,
              inputs=[f, mem.rho, mem.u],
              device="cuda")
    mem.export_final(prefix)
    wp.synchronize()
