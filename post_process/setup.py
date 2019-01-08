import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import subprocess
 
#proc_libs = subprocess.check_output("pkg-config --libs opencv".split())
#proc_incs = subprocess.check_output("pkg-config --cflags opencv".split())
#libs = [lib for lib in str(proc_libs, "utf-8").split()]

sourcefiles = [
"post_process.pyx",
]

libraries = [
"opencv_imgcodecs",
"opencv_imgproc",
"opencv_core"
]

include= [
numpy.get_include(), 
"/usr/local/include"
]

extensions = [
    Extension("post_process", sourcefiles,
        include_dirs = include,
        language="c++",
        libraries = libraries,
        library_dirs = ["/usr/local/lib", "/usr/lib/x86_64-linux-gnu"],
        extra_compile_args = ["-std=c++14","-DFLOAT32_DATA"],
        extra_link_args = ["-std=c++14"]        
    ),
]
setup(
    name = "PixelLinkPostProcess",
    ext_modules = cythonize(extensions),
)
