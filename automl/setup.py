from setuptools import setup
from setuptools import find_packages

setup(
    name='automl', #project name
    version='0.1.0',
    description='AutoML',
    #url
    author='Brando Miranda',
    author_email='miranda9@illinois.edu',
    license='MIT',
    packages=find_packages(),
    install_requires=['torch','numpy','scikit-learn','scipy','matplotlib','pyyml']
)

#install_requires=['numpy>=1.11.0']
