from setuptools import setup

setup(name='geoboed',
      version='0.1',
      description='GeoBED',
      author='Dominik Strutz',
      author_email='dominik.strutz@ed.ac.uk',
      license='MIT',
      packages=['geoboed',
                'geoboed.core',
                'geoboed.functionmode',
                'geoboed.fwd_collection',
                'geoboed.samplemode'],
      # package_dir={'geobed'},
      # install_requires=[
      #       'numpy',
      #       'torch',
      #       'pyro-ppl',
      #       'tqdm',
      #       'scikit-learn',
      # ],
      # extras_require={
      #       'h5py',
      #       'matplotlib',
      #       'pykonal'
      # }
      )