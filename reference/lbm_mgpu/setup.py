import typing
import warp as wp
import lbm_mgpu


def setup_LDC_problem(params, partitions: lbm_mgpu.Partition, mem):
    # Initialize the kernels
    kernels = lbm_mgpu.Kernels(params, mem)

    for i, p in enumerate(partitions):
        # Set the boundary conditions
        shape = p.shape
        wp.launch(kernels.get_set_lid_problem(5),
                  dim=shape,
                  inputs=[p, mem.bc_type[i]],
                  device=params.gpus[i])
        wp.launch(kernels.get_set_f_to_equilibrium(),
                  dim=shape,
                  inputs=[p, mem.bc_type[i], mem.f_0[i]],
                  device=params.gpus[i])
        pass
    wp.synchronize()


def export_final(prefix, params, partitions: lbm_mgpu.Partition, mem, f):
    # Compute the macroscopic variables
    wp.synchronize()
    for i, p in enumerate(partitions):
        kernels = lbm_mgpu.Kernels(params, mem)
        wp.launch(kernels.get_macroscopic(),
                  dim=p.shape,
                  inputs=[p, f[i], mem.rho[i], mem.u[i]],
                  device=params.gpus[i])
    mem.export_final(prefix)
    wp.synchronize()

def export_setup(prefix, params, partitions: lbm_mgpu.Partition, mem):
    # Compute the macroscopic variables
    wp.synchronize()
    mem.export_problem_setup(prefix)
    wp.synchronize()