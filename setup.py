from setuptools import setup, find_packages

setup(name='SpellvarDetection',
      version='0.1.0',
      url='https://github.com/fab-bar/SpellvarDetection',
      author='Fabian Barteld',
      author_email='fabian.barteld@rub.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy<1.15',
          'scipy',
          'sklearn',
          'imblearn'
      ])
