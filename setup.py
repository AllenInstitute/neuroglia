import setuptools
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

setuptools.setup(
    name="neuroglia",
    version="0.2.3",
    url="https://github.com/AllenInstitute/neuroglia",

    author="Justin Kiggins",
    author_email="justink@alleninstitute.org",

    description="scikit-learn compatible transformers for neural data science",

    packages=setuptools.find_packages(),

    ext_modules=cythonize(
        [
            Extension(
                "neuroglia.calcium.oasis.oasis_methods",
                sources=["neuroglia/calcium/oasis/oasis_methods.pyx"],
                include_dirs=[np.get_include()],
                language="c++",
            ),
        ],
        compiler_directives={'cdivision': True},
        ),

    install_requires=[
        'pandas',
        'xarray',
        'scikit-learn',
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
