import sc2env.data
from sc2env.sc2env import SC2Env
from gym.envs.registration import register
import gym

register(
    id='SC2MoveToBeacon-v0',
    entry_point='sc2env:SC2Env',
    kwargs={
  "map_name": "MoveToBeacon",
  "feature_screen": 86,
  "feature_minimap": 64
}
)

register(
    id='SC2DefeatRoaches-v0',
    entry_point='sc2env:SC2Env',
    kwargs={
  "map_name": "DefeatRoaches",
  "feature_screen": 86,
  "feature_minimap": 64
}
)

def make(game, **kwargs):
    # from sc2env.sc2env import SC2Env
    # settings = data.get_settings(game)
    # settings.update(kwargs)
    #
    # return SC2Env(**settings)
    return gym.make(game)
