from setuptools import setup

setup(
    name='sc2env',
    version='0.0.1',
    install_requires=['baselines', 'gym', 'pysc2', 'numpy', 'cloudpickle', 'pyyaml'],
)

import pkg_resources

tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 1.4'
