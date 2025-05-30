import time

from PIL.SpiderImagePlugin import iforms

import lbm_mgpu
import warp as wp

exercise_name = "04_mgpu_ooc_seq"


# define main function
def main():
    debug = False
    wp.clear_kernel_cache()
    gpus = wp.get_cuda_devices()
    if len(gpus) == 1:
        gpus = gpus * 4
        # Initialize the parameters
    params = lbm_mgpu.Parameters(num_steps=600,
                                 gpus=gpus,
                                 nx=1024 // 1,
                                 ny=768 // 1,
                                 prescribed_vel=0.5,
                                 Re=10000.0)

    partitions = []

    for i in range(params.num_gpsu):
        partition = lbm_mgpu.Partition()
        partition.id = i
        partition.num_partitions = params.num_gpsu
        partition.slices_per_partition = params.dim[0] // params.num_gpsu
        partition.origin[0] = i * partition.slices_per_partition
        partition.origin[1] = 0

        partition.shape[0] = partition.slices_per_partition
        partition.shape[1] = params.dim[1]
        partition.shape_domain[0] = params.dim[0]
        partition.shape_domain[1] = params.dim[1]

        partition.shape_with_halo[0] = partition.shape[0] + 2
        partition.shape_with_halo[1] = partition.shape[1]  # Add halo in y direction

        partition.shape_green[0] = partition.shape[0]
        partition.shape_green[1] = partition.shape[1] - 2  # Add halo in y direction

        partition.shape_red[0] = partition.shape[0]
        partition.shape_red[1] = 2  # Add halo in y direction

        partitions.append(partition)

    def get_fields(partitions):
        fields = []
        for i, partition in enumerate(partitions):
            nx = partition.shape_with_halo[0]
            ny = partition.shape_with_halo[1]
            f = wp.zeros((params.Q, nx, ny), dtype=wp.float64, device=params.gpus[i])
            fields.append(f)
        return fields

    f_0 = get_fields(partitions)
    f_1 = get_fields(partitions)

    @wp.func
    def read_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32):
        return field[card, xi + 1, yi]

    @wp.func
    def write_field(field: wp.array3d(dtype=wp.float64), card: wp.int32, xi: wp.int32, yi: wp.int32,
                    value: wp.float64):
        field[card, xi + 1, yi] = value

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
    def green(
            partition: lbm_mgpu.Partition,
            omega: wp.float64,
            f_in: wp.array3d(dtype=wp.float64),
            bc_type_field: wp.array2d(dtype=wp.uint8),
            f_out: wp.array3d(dtype=wp.float64),
    ):
        # Get the global index
        it, jt = wp.tid()
        jt = jt + 1
        partition_index = wp.vec2i(it, jt)
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
            f_post = compute_boundaries(bc_type)

        mcrpc = compute_macroscopic(f_post)

        # Compute the equilibrium
        f_eq = compute_equilibrium(mcrpc)

        f_post = compute_collision(f_post, f_eq, mcrpc, omega)

        # Set the output
        for q in range(params.Q):
            write_field(field=f_out, card=q, xi=partition_index[0], yi=partition_index[1], value=f_post[q])

    # ---------------------------------------------------------

    @wp.kernel
    def red(
            partition: lbm_mgpu.Partition,
            omega: wp.float64,
            f_in: wp.array3d(dtype=wp.float64),
            bc_type_field: wp.array2d(dtype=wp.uint8),
            f_out: wp.array3d(dtype=wp.float64),
    ):
        # Get the global index
        it, jt = wp.tid()
        if jt == 1:
            jt = partition.shape[1] - 1
        partition_index = wp.vec2i(it, jt)
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
            f_post = compute_boundaries(bc_type)

        mcrpc = compute_macroscopic(f_post)

        # Compute the equilibrium
        f_eq = compute_equilibrium(mcrpc)

        f_post = compute_collision(f_post, f_eq, mcrpc, omega)

        # Set the output
        for q in range(params.Q):
            write_field(field=f_out, card=q, xi=partition_index[0], yi=partition_index[1], value=f_post[q])

    # ---------------------------------------------------------

    lbm_mgpu.setup_LDC_problem(params=params, partitions=partitions, mem=mem)
    lbm_mgpu.export_final(prefix=exercise_name, params=params, partitions=partitions, mem=mem, f=mem.f_0)
    lbm_mgpu.export_setup(prefix=exercise_name, params=params, partitions=partitions, mem=mem)

    # #mem.save_magnituge_vtk(0)
    def iterate():
        for i, p in enumerate(partitions):
            for q in range(params.Q):
                if i != params.num_gpsu - 1:
                    src = mem.f_0[i + 1][q, 1]
                    dst = mem.f_0[i][q, partition.shape_with_halo[0] - 1]
                    wp.copy(src=src, dest=dst, count=p.shape[1])
                if i != 0:
                    src = mem.f_0[i - 1][q, partition.shape_with_halo[0] - 2]
                    dst = mem.f_0[i][q, 0]
                    wp.copy(src=src, dest=dst, count=p.shape[1])
        wp.synchronize()

        for i, p in enumerate(partitions):
            wp.launch(green,
                      dim=p.shape_green,
                      inputs=[p, params.omega, mem.f_0[i], mem.bc_type[i], mem.f_1[i]],
                      device=params.gpus[i])

        for i, p in enumerate(partitions):
            wp.launch(red,
                      dim=p.shape_green,
                      inputs=[p, params.omega, mem.f_0[i], mem.bc_type[i], mem.f_1[i]],
                      device=params.gpus[i])

        wp.synchronize()
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

    lbm_mgpu.export_final(prefix=exercise_name, params=params, partitions=partitions, mem=mem, f=mem.f_0)

    # Statistics
    elapsed_time = stop - start
    mlups = params.compute_mlups(elapsed_time)
    print(f"Main loop time: {elapsed_time:5.3f} seconds")
    print(f"MLUPS:          {mlups:5.1f}")


# call the main when the script is called
if __name__ == "__main__":
    main()
