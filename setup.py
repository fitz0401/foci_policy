from setuptools import setup, find_packages

setup(
    name='foci_policy',
    version='1.0',
    packages=find_packages(),
    description='Official implementation of FOCI Policy',
    url='git@github.com:fitz0401/foci_policy.git',
    author='ze fu',
    author_email='ze.fu@kuleuven.be',
    license='MIT',
    install_requires=[
        'typing_extensions',
    ],
    zip_safe=False
)
