import setuptools
from setuptools import setup, find_packages


setup(
    name='icocr',
    version='0.0.2',  # Ideally, a semantic versioning number
    packages=find_packages(),
    install_requires=[
        # List of your project dependencies
        # 'opencv-python',
        # 'onnxruntime-gpu',
        # 'numpy',
        # 'python-math',
        # 'scikit-image',
        # 'Pillow',
        # 'PyYAML',
        # 'scipy'
    ],
    author='Thai Tran',
    author_email='thai.tran1@icomm-group.com',
    description='A short description of my package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/iC-RnD/icocr-vi-en.git',  # if exists
    classifiers=[
        # Classifiers help categorize your project
        # Full list: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # example license
        # ...
    ],
    # ... other optional arguments
)