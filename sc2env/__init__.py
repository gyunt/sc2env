from sc2env.sc2env import SC2Env


from gym.envs.registration import register

register(
    id='SC2MoveToBeacon-v0',
    entry_point='sc2env:SC2Env',
    kwargs={'map_name': 'MoveToBeacon'}
)
__ALL__ = [SC2Env]