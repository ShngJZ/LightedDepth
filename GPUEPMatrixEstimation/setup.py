from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'GPUEPM'
sources = [join(project_root, file) for file in [
    "essential_matrix.cu",
    "epm_wrapper.cpp"
]]

setup(
    name='GPUEPM',
    ext_modules=[
        CUDAExtension('GPUEPM',
            sources), # extra_compile_args, extra_link_args
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

