#Setup.py helps to distribute your python package, using pip
#It will contain author name, version of the package, dependencies that the package requires
#With this we can use simple pip command to install the package and import it any location
#It automatically builds the package and install required dependencies

from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path : str) -> List[str]:
  '''
  return list of requirements for the project
  '''

  with open(file_path) as file_obj:
    requirements = file_obj.readlines()
    requirements = [r.replace("\n","") for r in requirements]
    if HYPHEN_E_DOT in requirements:
      requirements.remove(HYPHEN_E_DOT)
    return requirements





setup(

  name = 'end-to-end ml project',
  version='0.0.1',
  author= 'Rajadurai',
  author_email='rajadurai3491@gmail.com',
  packages=find_packages(),
  install_requires=get_requirements('requirements.txt')

)