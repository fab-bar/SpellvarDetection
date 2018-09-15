from setuptools import setup, find_packages

setup(name='SpellvarDetection',
      use_scm_version=True,
      url='https://github.com/fab-bar/SpellvarDetection',
      author='Fabian Barteld',
      author_email='fabian.barteld@rub.de',
      license='MIT',
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=[
          'numpy<1.15',
          'scipy',
          'sklearn',
          'imbalanced-learn',
          'spsim',
          'flask'
      ])
