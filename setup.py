from setuptools import setup

setup(name='geobed',
      version='0.1',
      description='GeoBED',
      author='Dominik Strutz',
      author_email='dominik.strutz@ed.ac.uk',
      license='MIT',
      # packages=['geobed',
      #           'geobed.fwd_collection.avo',
      #           'geobed.fwd_collection.raytracer'
      #           'geobed.fwd_collection.raytracer_improved',
      #           'geobed.fwd_collection.pyprop2pytorch',
      #           'geobed.continuous.core',
      #           'geobed.discrete.core',
      #           'geobed.discrete.design2data_helpers',
      #           'geobed.discrete.eig',
      #           'geobed.discrete.optim',
      #           'geobed.discrete.utils',
      # ],         
      package_dir={},
      install_requires=[
            'numpy',
            'torch',
            'tqdm',
            'h5py',
            'matplotlib',
            'dill',
            'multiprocess',
            'mpire',
            'joblib',
            'zuko',
      ],
      extras_require={
            # 'plotting':   'matplotlib',
            'raytracing': ['pykonal', 'ttcrpy', 'vtk'], # vtk necessary for ttrcpy but not installed by default
            'pyprop8':    'pyprop8',
            'full':      ['pykonal', 'ttcrpy', 'vtk', 'segyio', 'seaborn'], # for convenience if I make a new environment
            }
      )
