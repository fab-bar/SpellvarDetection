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
          'click',
          'numpy',
          'scipy',
          'scikit-learn',
          'imbalanced-learn',
          'spsim',
          'flask',
          'tinymongo'
      ],
      entry_points={
        'console_scripts': ['spellvardetection = spellvardetection.cli:main']
      }
)
