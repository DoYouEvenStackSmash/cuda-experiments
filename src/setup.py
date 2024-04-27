import os
import sys
import setuptools
from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension
class get_pybind_include(object):
    def __str__(self):
        import pybind11

        return pybind11.get_include()

ext_modules = [
    Extension("mult_example", ["numpy_multiply.cpp"],include_dirs=[get_pybind_include()],language="c++",
),
]

setup(
    name="mult_example",
    cmdclass={"build_ext": build_ext},
    ext_modules=ext_modules,
    zip_safe=False,

)
