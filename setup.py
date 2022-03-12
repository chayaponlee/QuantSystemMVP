import setuptools

"""
Run 'pip install -e.' in the root project folder to install quantlib library 
"""

setuptools.setup(

    name='realgam',
    version=0.1,
    description='Personal quant library/tools',
    url='#',
    author='Realgam',
    install_requires=['opencv-python'],
    author_email='',
    packages=['realgam'],
    zip_safe=False


)