from setuptools import setup, find_packages

setup(
    name='textentlib',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'click',
        'dask',
        'spacy',
    ],
    entry_points={
        'console_scripts': [
            'tei2spacy=tei2spacy:main',
        ],
    },
)