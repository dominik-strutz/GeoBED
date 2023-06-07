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
            'matplotlib', # could be optional but it is so common that it is not worth it
            'dill',
            'mpire',
            'joblib',
      ],
      extras_require={
            # 'plotting':   'matplotlib',
            'raytracing': ['pykonal', 'ttcrpy', 'vtk'], # vtk is needed for ttrcpy but not automatically installed
            'pyprop8':    'pyprop8',
            'full': ['pykonal', 'ttcrpy', 'vtk', 'nb_conda_kernels', 'ipykernel', 'segyio', 'pyprop8']
      }
      )
