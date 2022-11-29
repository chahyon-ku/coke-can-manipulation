import os
import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
from torch.utils.cpp_extension import load


def load_cpp_ext():
    ext_name = "knn_pytorch"
    root_dir = os.path.join(os.path.split(__file__)[0])
    ext_csrc = os.path.join(root_dir, "src")
    ext_path = os.path.join(root_dir, "build")
    os.makedirs(ext_path, exist_ok=True)
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False."
    ext_sources = [
        os.path.join(ext_csrc, "knn_pytorch.cpp"),
        os.path.join(ext_csrc, "knn_cuda_kernel.cu")
    ]
    extra_cuda_cflags = [
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    ext = load(
        name=ext_name,
        sources=ext_sources,
        extra_cflags=["-O2"],
        build_directory=ext_path,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
        with_cuda=True
    )
    return ext


_knn = load_cpp_ext()


class KNearestNeighbor(torch.nn.Module):
    """ Compute k nearest neighbors for each query point.
    """
    def __init__(self, k):
        super(KNearestNeighbor, self).__init__()
        self.k = k

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), "ref.shape={} != query.shape={}".format(ref.shape, query.shape)
        with torch.no_grad():
            batch_size = ref.size(0)
            # D, I = [], []
            I = []
            for bi in range(batch_size):
                r, q = ref[bi], query[bi]
                d, i = _knn.knn(r.float(), q.float(), self.k)
                # D.append(d)
                I.append(i)
            # D = torch.stack(D, dim=0)
            I = torch.stack(I, dim=0)
        return I


class TestKNearestNeighbor(unittest.TestCase):

    def test_forward(self):
        knn = KNearestNeighbor(2)
        while (1):
            D, N, M = 128, 100, 1000
            ref = Variable(torch.rand(2, D, N))
            query = Variable(torch.rand(2, D, M))

            inds = knn(ref, query)
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
            # ref = ref.cpu()
            # query = query.cpu()
            print(inds)


if __name__ == '__main__':
    unittest.main()
