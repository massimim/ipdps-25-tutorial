import warp as wp

@wp.struct
class Macroscopic:
    rho: wp.float64
    u: wp.vec(length=2, dtype=wp.float64)
