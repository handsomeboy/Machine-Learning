from setuptools import setup
from Cython.Build import cythonize
setup(
    name = 'My prime module',
    ext_modules = cythonize("learnmultilayer.pyx"),
)