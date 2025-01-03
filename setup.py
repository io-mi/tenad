from setuptools import setup, find_packages

setup(
    name="tenad",
    version="0.0.1",
    author="Rizo Hashimi",
    packages=find_packages(".", include=["tenad"]),
    install_requires=["numpy>=2.2.1"]
)
