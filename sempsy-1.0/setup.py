
from setuptools import setup,find_packages

setup(
    name="sempsy",
    version="1.0",
    author="Li Fan",
    author_email="lfan.space@gmail.com",
    description="Compute semantic distance simply",
    url="https://github.com/semantics2psycho/Semantic-Toolbox",
    packages=['textprocess','textprocess.tests','semdistance','semdistance.tests','corpustrain']
    # packages = find_packages()
)
