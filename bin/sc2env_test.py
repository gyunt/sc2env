import logging
import sys
from collections import defaultdict

import gym

from sc2env import SC2Env

logger_pysc2 = logging.getLogger('pysc2')
logger_pysc2.setLevel(logging.DEBUG)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(message)s'))
logger_pysc2.addHandler(stream_handler)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}

_game_envs['SC2'] = {
    'SC2MoveToBeacon-v0',
    'SC2MoveToBeacon-v1',
    'SC2CollectMineralShards-v0',
    'SC2CollectMineralShards-v1',
    'SC2CollectMineralShards-v2',
    'SC2FindAndDefeatZerglings-v0',
    'SC2DefeatRoaches-v0',
    'SC2DefeatZerglingsAndBanelings-v0',
    'SC2CollectMineralsAndGas-v0',
    'SC2BuildMarines-v0',
}


def main(_):
    env = SC2Env('SC2MoveToBeacon-v0')

    while True:
        obs, rew, done, _ = env.step()

        if done:
            obs = env.reset()


if __name__ == '__main__':
    # runner.run(main, 'mhp', args=(sys.argv), host='127.0.0.1', port=8000)
    main(sys.argv)
