import setuptools

setuptools.setup(
    name="neuroglia",
    version="0.1.0",
    url="https://github.com/AllenInstitute/neuroglia",

    author="Justin Kiggins",
    author_email="justink@alleninstitute.org",

    description="pipelines for transformations for neurophysiology data",
    long_description=open('README.rst').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
