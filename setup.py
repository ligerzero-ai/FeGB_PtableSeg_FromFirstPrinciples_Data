from setuptools import setup, find_packages

setup(
    name="FeGB_PtableSeg",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pymatgen==2023.10.11",
        "ipython",
        "jupyter"
    ],
    python_requires='>=3.9',
)
