from setuptools import setup, find_packages

setup(
    name = 'finefood',
    version = '0.1.0',
    url = 'https://github.com/jakeret/finefood.git',
    author = 'Joel Akeret',
    author_email = 'joel.akeret@gmail.com',
    description = 'Amazon Fine Food and Polyaxon',
    packages = find_packages(),
    package_dir={'finefood': 'finefood'},
    install_requires = [],
)