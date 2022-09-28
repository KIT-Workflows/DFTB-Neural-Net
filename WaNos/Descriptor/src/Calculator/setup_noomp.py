import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

os.environ['CC'] = 'gcc'


#setup(name='SymmFuncIndCython',
#      ext_modules=cythonize("SymmFuncIndCython.pyx"))


#from Cython.Compiler.Options import directive_defaults

#directive_defaults['linetrace'] = True
#directive_defaults['binding'] = True

# SymmFuncIndCAPIModule= Extension('SymmFuncIndCAPI',
#                                     sources = ['SymmFuncIndCAPI.c'],
#                                     include_dirs = [python_include, np_include],
#                                     libraries=['m', 'python3.6m'],
#                                     library_dirs=[conda_lib_dir, np_dir]
#                                     )


SymmDerivIndCythonMod = [
Extension("SymmDerivIndCython",
          ['SymmDerivIndCython.pyx'],
          extra_compile_args = [],
          extra_link_args=[])
]

SymmFuncIndCythonMod = [
Extension("SymmFuncIndCython",
          ['SymmFuncIndCython.pyx'],
          extra_compile_args = [],
          extra_link_args = [])
]

SymmFuncIndVecCythonMod = [
Extension("SymmFuncIndVecCython",
          ['SymmFuncIndVecCython.pyx'],
          extra_compile_args = [],
          extra_link_args = [])
]

SymmDerivIndVecCythonMod = [
Extension("SymmDerivIndVecCython",
          ['SymmDerivIndVecCython.pyx'],
          extra_compile_args = [],
          extra_link_args = [])
]


setup(name='SymmDerivIndCython',
     ext_modules=cythonize(SymmDerivIndCythonMod),
     include_dirs=[np.get_include()]
)

setup(name='SymmFuncIndCython',
     ext_modules=cythonize(SymmFuncIndCythonMod),
     include_dirs=[np.get_include()]
)

setup(name='SymmDerivIndVecCython',
    ext_modules=cythonize(SymmDerivIndVecCythonMod),
    include_dirs=[np.get_include()]
)

setup(name='SymmFuncIndVecCython',
    ext_modules=cythonize(SymmFuncIndVecCythonMod),
    include_dirs=[np.get_include()]
)

