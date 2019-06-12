from setuptools import setup, find_packages
import versioneer


setup(name='SpellvarDetection',
      url='https://github.com/fab-bar/SpellvarDetection',
      author='Fabian Barteld',
      author_email='fabian.barteld@rub.de',
      license='MIT',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      packages=find_packages(),
      setup_requires=[],
      install_requires=[
          'factory-manager',
          'click',
          'joblib',
          'jsonpickle',
          'numpy',
          'scipy',
          'scikit-learn',
          'imbalanced-learn',
          'tensorflow',
          'spsim',
          'flask',
          'tinymongo'
      ],
      entry_points={
        'console_scripts': ['spellvardetection = spellvardetection.cli:main']
      }
)
