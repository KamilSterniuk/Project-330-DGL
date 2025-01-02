from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='spmm_extension',
    ext_modules=[
        CppExtension(
            name='spmm_extension',
            sources=['spmm_extension.cpp'],
            extra_compile_args=['-fopenmp'],  # flaga dla OpenMP
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
