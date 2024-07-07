from setuptools import setup, find_packages

setup(name='BUSDENSITY',
      version=0.1,
      author='Arianna Bunnell',
      author_email='abunnell@hawaii.edu',
      description='Deep learning model predicting BI-RADS mammographic breast density from BUS',
      packages=find_packages(),
      license='cc-by-nc-sa 4.0',
      include_package_data=True,
      install_requires=[
          'numpy >= 1.23.5', 'pandas >= 1.4.4', 'optuna >= 3.1.0', 'scikit-learn >= 1.2.1',
          'pillow >= 9.3.0', 'torchvision >= 0.13.1', 'lightning >= 1.9.4'
      ])