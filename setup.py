from setuptools import find_packages,setup
from typing import List

hypen_dot = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This fucntion will retutn list of requirements
    '''
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n',' ') for req in requirements]

    if hypen_dot in requirements:
        requirements.remove(hypen_dot)    



setup(
    name = 'ML - 2',
    version='0.0.1',
    author='Jyothi kumar',
    author_email='jyothikumargoud6@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
