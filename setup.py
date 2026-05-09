from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->list[str]:
    '''
    this function is meant to read the requirements.txt file and return the package names in list formate
    without any next line escape charater which may confuse the installation command in setup file.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements

setup(

    name='FirstProfessionalMLProject',
    version='1.0',
    author='Rising Keshav',
    author_email='keshav48kumar@gmail.com',
    maintainer='Keshav Kumar',
    description=" this is the first Industry standard ML project ",
    long_description="This Project will be my yard stick and this is how i will create my future projects. This is going to be the basic structure of my project setup",
    packages=find_packages(),
    # install_requires=['pandas','numpy','seaborn'] this is not professional way
    install_requires=get_requirements('requirements.txt')
)