import typing
import warp as wp
import lbm_mgpu
from numpy.ma.core import shape


class Stream:
    def __init__(self, parameters: lbm_mgpu.Parameters,
                 read_field):
        self.params = parameters
        self.read_field = read_field

    def get_pull_stream(self):
        """
        Get a warp function to pull the stream from the previous timestep.
        """
        Q = self.params.Q
        D = self.params.D
        shape_dev = self.params.shape_dev
        c_dev = self.params.c_dev
        read_field = self.read_field

        @wp.func
        def pull_stream(
                index: typing.Any,
                f_mem: wp.array3d(dtype=wp.float64),
        ):

            f_vec = wp.vec(length=Q, dtype=wp.float64)

            for q in range(Q):
                pull_ngh = wp.vec2i(0, 0)
                outside_domain = False
                for d in range(D):
                    pull_ngh[d] = index[d] - c_dev[d, q]

                    if pull_ngh[d] < 0 or pull_ngh[d] >= shape_dev[d]:
                        outside_domain = True
                if not outside_domain:
                    f_vec[q] = read_field(f_mem, q, pull_ngh[0], pull_ngh[1])

            return f_vec

        return pull_stream

