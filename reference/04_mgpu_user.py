import time

from PIL.SpiderImagePlugin import iforms

import lbm_mgpu
import warp as wp

exercise_name = "03_fusion_user"


# define main function
def main():
    debug = False
    wp.clear_kernel_cache()
    gpus = wp.get_cuda_devices()
    if len(gpus) == 1:
        gpus = gpus * 2
        # Initialize the parameters
    params = lbm_mgpu.Parameters(num_steps=50000,
                                 gpus=gpus * 2,
                                 nx=1024,
                                 ny=768,
                                 prescribed_vel=0.5,
                                 Re=10000.0)

    partitions = [lbm_mgpu.Partition] * len(gpus)

    for i, partition in enumerate(partitions):
        partition.id = i
        partition.num_partitions = len(partitions)
        partition.slices_per_partition = params.domain_shape[1] // len(partitions)
        partition.origin = i * partition.slices_per_partition

        partition.shape = params.domain_shape
        partition.shape[1] = partition.slices_per_partition
        partition.shape_domain = params.domain_shape

        partition.shape_with_halo = partition.shape
        partition.shape_with_halo[2] = partition.shape[2] + 2  # Add halo in y direction

    def get_fields(partitions):
        fields = []
        for i, partition in enumerate(partitions):
            f = wp.zeros((params.Q,) + partition.shape_with_halo, dtype=wp.float64)
            fields.append(f)
        return fields

    f_0 = get_fields(partitions)
    f_1 = get_fields(partitions)

    @wp.func
    def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
        return field[card, xi, yi + 1]

    @wp.func
    def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                    value: wp.float64):
        field[card, xi, yi + 1] = value

    # Initialize the memory
    mem = lbm_mgpu.Memory(params,
                          partitions,
                          f_0=f_0,
                          f_1=f_1,
                          read=read_field,
                          write=write_field)

    # Initialize the kernels
    functions = lbm_mgpu.Functions(params)
    kernels = lbm_mgpu.Kernels(params, mem)

    Q = params.Q
    D = params.D
    bc_bulk = params.bc_bulk
    c_dev = params.c_dev

    compute_boundaries = functions.get_apply_boundary_conditions()

    compute_macroscopic = functions.get_macroscopic()
    compute_equilibrium = functions.get_equilibrium()
    compute_collision = functions.get_kbc()

    # ---------------------------------------------------------

    @wp.kernel
    def fused(
            partition: lbm_mgpu.Partition,
            omega: wp.float64,
            f_in: wp.array3d(dtype=wp.float64),
            bc_type_field: wp.array2d(dtype=wp.uint8),
            f_out: wp.array3d(dtype=wp.float64),
    ):
        # Get the global index
        i, j = wp.tid()
        partition_index = wp.vec2i(i, j)
        domain_index = partition_index + partition.origin

        f_post = wp.vec(length=Q, dtype=wp.float64)
        bc_type = bc_type_field[partition_index[0], partition_index[1]]

        for q in range(params.Q):
            partition_pull_ngh = wp.vec2i(0, 0)
            domain_pull_ngh = wp.vec2i(0, 0)

            outside_domain = False

            for d in range(D):
                partition_pull_ngh[d] = partition_index[d] - c_dev[d, q]
                domain_pull_ngh[d] = domain_index[d] - c_dev[d, q]
                if domain_pull_ngh[d] < 0 or domain_pull_ngh[d] >= partition.shape_domain[d]:
                    outside_domain = True
            if not outside_domain:
                f_post[q] = read_field(field=f_in, card=q, xi=partition_pull_ngh[0], yi=partition_pull_ngh[1])

        if bc_type != bc_bulk:
            f_post = compute_boundaries(partition, bc_type)

        mcrpc = compute_macroscopic(f_post)

        # Compute the equilibrium
        f_eq = compute_equilibrium(mcrpc)

        f_post = compute_collision(f_post, f_eq, mcrpc, omega)

        # Set the output
        for q in range(params.Q):
            write_field(field=f_out, card=q, xi=partition_index[0], yi=partition_index[1], value=f_post[q])

    # ---------------------------------------------------------
    lbm_mgpu.setup_LDC_problem(params=params, partitions=partitions, mem=mem)

    # #mem.save_magnituge_vtk(0)
    def iterate():
        wp.launch(fused,
                  dim=params.grid_shape,
                  inputs=[params.omega, mem.f_0, mem.bc_type, mem.f_1],
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

    lbm.export_final(prefix=exercise_name, params=params, mem=mem, f=mem.f_0)

    # Statistics
    elapsed_time = stop - start
    mlups = params.compute_mlups(elapsed_time)
    print(f"Main loop time: {elapsed_time:5.3f} seconds")
    print(f"MLUPS:          {mlups:5.1f}")


# call the main when the script is called
if __name__ == "__main__":
    main()
