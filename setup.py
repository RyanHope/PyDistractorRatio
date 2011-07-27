from numpy.distutils.core import setup, Extension

module1 = Extension('perception', sources = ['perception.c'])

setup (name = 'Perception',
        version = '1.0',
        ext_modules = [module1])