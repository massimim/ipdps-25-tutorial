import warp as wp


@wp.struct
class Partition:
    id: wp.int32
    num_partitions: wp.int32
    slices_per_partition: wp.int32
    origin: wp.vec(length=2, dtype=wp.int32)
    shape: wp.vec(length=2, dtype=wp.int32)
    shape_with_halo: wp.vec(length=2, dtype=wp.int32)
    shape_domain: wp.vec(length=2, dtype=wp.int32)
