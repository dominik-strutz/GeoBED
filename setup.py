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
            'matplotlib'
      ],
      extras_require={
            'plotting':   'matplotlib',
            'raytracing': 'pykonal',
            'pyprop8':    'pyprop8'
      }
      )
