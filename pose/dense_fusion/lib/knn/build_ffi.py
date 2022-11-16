# https://gist.github.com/tonyseek/7821993
import glob
import os

import torch
from os import path as osp
from torch.utils.cpp_extension import load

abs_path = osp.dirname(osp.realpath(__file__))
extra_objects = [osp.join(abs_path, 'build/knn_cuda_kernel.so')]
extra_objects += glob.glob('/usr/local/cuda/lib64/*.a')


def load_cpp_ext(ext_name):
    ext_name = "knn_pytorch"
    root_dir = os.path.join(os.path.split(__file__)[0])
    ext_csrc = os.path.join(root_dir, "src")
    ext_path = os.path.join(root_dir, "build", ext_name)
    os.makedirs(ext_path, exist_ok=True)
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False."
    ext_sources = [
        os.path.join(ext_csrc, "cuda", "knn_pytorch.c"),
        os.path.join(ext_csrc, "cuda", "knn_cuda_kernel.cu")
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


ffi = create_extension(
    'knn_pytorch',
    headers=['src/knn_pytorch.h'],
    sources=['src/knn_pytorch.c'],
    define_macros=[('WITH_CUDA', None)],
    relative_to=__file__,
    with_cuda=True,
    extra_objects=extra_objects,
    include_dirs=[osp.join(abs_path, 'include')]
)


if __name__ == '__main__':
    assert torch.cuda.is_available(), 'Please install CUDA for GPU support.'
    ffi.build()
