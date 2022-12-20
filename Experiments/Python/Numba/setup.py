from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("helloworld.pyx")
)


# extensions = cythonize(extensions, language_level = "3")
