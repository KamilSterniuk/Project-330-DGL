from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='spmm_csr_extension',
    ext_modules=[
        CppExtension('spmm_csr_extension', ['spmm_csr.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
