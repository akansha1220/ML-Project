from setuptools import find_namespace_packages,setup,find_packages
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirements
    '''
    requirements = []

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
name="MLProject",
version='0.0.2',
author='Akansha',
author_email='akansharathi121@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirement.txt')
)
    
