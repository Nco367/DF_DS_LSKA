#!/usr/bin/env python3
import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 强制设置环境变量
os.environ["CC"] = "gcc-11"
os.environ["CXX"] = "g++-11"

_ext_src_root = "./_ext-src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

this_dir = os.path.dirname(os.path.abspath(__file__))

print("Include path:", "{}/include".format(_ext_src_root))
print("Does it exist?", os.path.exists("{}/include".format(_ext_src_root)))
print("Header files:", os.listdir("{}/include".format(_ext_src_root)))

include_dirs = [
    os.path.join(this_dir, '_ext-src/include')
],

setup(
    name='df6d',
    ext_modules=[
        CUDAExtension(
            name='pointnet2_ops._ext',
            sources=_ext_sources,
            include_dirs=[
                os.path.join(this_dir, '_ext-src/include'),
            ],
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format(include_dirs)],
                "nvcc": ["-O2", "-I{}".format(include_dirs)],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


try:
    src_pth = './build'
    tg_pth = 'lib/pointnet2_utils/'
    fd_lst = os.listdir(src_pth)
    for fd in fd_lst:
        if 'lib' in fd:
            src_pth = os.path.join(src_pth, fd, 'pointnet2_utils')
            f_nm = os.listdir(src_pth)[0]
            src_pth = os.path.join(src_pth, f_nm)
            tg_pth = os.path.join(tg_pth, f_nm)
    os.system('cp {} {}'.format(src_pth, tg_pth))
    print(
        src_pth, '==>', tg_pth,
    )
except:
    print(
        "\n****************************************************************\n",
        "Failed to copy builded .so to ./pvn3d/lib/pointnet2_utils/.\n",
        "Please manually copy the builded .so file (_ext.cpython*.so) in ./build"+\
        " to ./pvn3d/lib/pointnet2_utils/.",
        "\n****************************************************************\n"
    )

# vim: ts=4 sw=4 sts=4 expandtab
