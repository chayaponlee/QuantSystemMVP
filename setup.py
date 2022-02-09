import setuptools

"""
Run 'pip install -e.' in the root project folder to install quantlib library 
"""

setuptools.setup(

    name='quantlib',
    version=0.1,
    description='Personal quant library/tools for Pun Lee',
    url='#',
    author='Realgam',
    install_requires=['opencv-python'],
    author_email='',
    packages=setuptools.find_packages(),
    zip_safe=False


)