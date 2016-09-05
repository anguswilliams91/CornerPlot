from setuptools import setup

setup(
    name="corner_plot",
    version="0.1",
    author="Angus Williams",
    author_email="anguswilliams91@gmail.com",
    packages=['corner_plot'],
    package_dir={'corner_plot':'src/corner_plot'},
    install_requires=['numpy','matplotlib']
    )