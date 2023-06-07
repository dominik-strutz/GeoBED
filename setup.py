from setuptools import setup

setup(name='geobed',
      version='0.1',
      description='GeoBED',
      author='Dominik Strutz',
      author_email='dominik.strutz@ed.ac.uk',
      license='MIT',
      packages=['geobed',
                'geobed.fwd_collection',],
      package_dir={},
      install_requires=[
            'numpy',
            'torch',
            'tqdm',
            'h5py',
            'matplotlib',
            'dill',
            'mpire',
            'joblib',
      ],
      extras_require={
            # 'plotting':   'matplotlib',
            'raytracing': ['pykonal', 'ttcrpy', 'vtk'], # vtk necessary for ttrcpy but not installed by default
            'pyprop8':    'pyprop8',
            'full':      ['pykonal', 'ttcrpy', 'vtk', 'segyio', 'seaborn'], # for convenience if I make a new environment
            }
      )
