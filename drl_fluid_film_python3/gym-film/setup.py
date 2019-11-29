from setuptools import setup, find_namespace_packages

setup(name='gym_film',
      version='1.0.1',
      packages=find_namespace_packages(),
      install_requires=['gym', 'stable_baselines', 'matplotlib', 'numpy', 'tensorflow',
                        'argparse']  # And any other dependencies foo needs
      )
# TODO - look at the dependencies, make sure to have all of them, and no unnecessary ones
# TODO - in general, check unecessary imports
