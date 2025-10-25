from setuptools import find_packages, setup

def get_requirements(file_path: str) -> list:
    """This function will return the list of requirements"""
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements if req.strip() and not req.startswith("-e")]
    return requirements

setup(
name="project-1",
version="0.1.0",
author="Aryan",
author_email="aryan78231@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.tst"),
)