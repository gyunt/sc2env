import argparse

import gym
import numpy as np
# noinspection PyUnresolvedReferences
import sc2env
from absl import flags
from pysc2.lib import actions
from pysc2.lib import features

FLAGS = flags.FLAGS
FLAGS([__file__])
FUNCTIONS = actions.FUNCTIONS

__description__ = 'Run a scripted example using the SC2MoveToBeacon-v0 environment.'

_PLAYER_ENEMY = features.PlayerRelative.ENEMY
_PLAYER_NEUTRAL = 3  # beacon/minerals
_NO_OP = 0

_ENV_NAME = "SC2MoveToBeacon-v0"


# _ENV_NAME = 'SC2DefeatRoaches-v0'
# _ENV_NAME = 'SC2FindAndDefeatZerglings-v0'


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class MoveToBeacon1d(object):
    def __init__(self, env_name, visualize=False, step_mul=None, random_seed=None):
        super().__init__()
        self.env_name = env_name
        self.visualize = visualize
        self.step_mul = step_mul
        self.random_seed = random_seed

    def run(self, num_episodes=1000):

        env = gym.make(self.env_name)
        env.settings['visualize'] = self.visualize
        env.settings['step_mul'] = self.step_mul
        env.settings['random_seed'] = self.random_seed
        env.settings['map_name'] = 'MoveToBeacon'
        env.flatten_action = False
        env.settings['realtime'] = True

        episode_rewards = np.zeros((num_episodes,), dtype=np.int32)
        episodes_done = 0
        for ix in range(num_episodes):
            obs = env.reset()

            done = False
            while not done:
                action = self.get_action(env, obs)
                obs, reward, done, _ = env.step(action)
                self.print_interesting_actions(env, obs)

            # stop if the environment was interrupted for any reason
            if obs is None:
                break

            episodes_done += 1
            episode_rewards[ix] = env.episode_reward

        env.close()

        return episode_rewards[:episodes_done]

    def get_action(self, env, obs):
        interested_action = [7, 331]
        available_actions = \
            [action for action in env.available_actions if action in interested_action]

        print(env.available_actions)

        function_id = np.random.choice(available_actions)
        args = []
        for arg in env.action_spec[0].functions[function_id].args:
            if arg.name == 'queued':
                a = [0]
            else:
                a = [np.random.randint(0, size) for size in arg.sizes]
            args.append(a)

        ret = [function_id] + args
        # print(ret)
        return ret

    # class DefeatRoaches(base_agent.BaseAgent):
    #     """An agent specifically for solving the DefeatRoaches map."""
    #
    #     def step(self, obs):
    #         super(DefeatRoaches, self).step(obs)
    #         if FUNCTIONS.Attack_screen.id in obs.observation.available_actions:
    #             player_relative = obs.observation.feature_screen.player_relative
    #             roaches = _xy_locs(player_relative == _PLAYER_ENEMY)
    #             if not roaches:
    #                 return FUNCTIONS.no_op()
    #
    #             # Find the roach with max y coord.
    #             target = roaches[numpy.argmax(numpy.array(roaches)[:, 1])]
    #             return FUNCTIONS.Attack_screen("now", target)
    #
    #         if FUNCTIONS.select_army.id in obs.observation.available_actions:
    #             return FUNCTIONS.select_army("select")
    #
    #         return FUNCTIONS.no_op()
    def print_interesting_actions(self, env, obs):
        intereseted = [0,  # "no_op"
                       2,  # "select_point"
                       # 3,  # "select_rect"
                       # 4,  # "select_control_group"
                       # 5,  # "select_unit"
                       # 12,  # "attack_screen"
                       14,  # "attack_attack_screen"
                       140,  # "Cancel_quick"
                       # 261,  # "Halt_quick"
                       # 264,  # "Harvest_Gather_screen"
                       # 269,  # "Harvest_Return_quick"
                       274,  # "HoldPosition_quick"
                       # 275,  # "Land_screen"
                       # 281,  # "Lift_quick"
                       # 287,  # "Load_screen" 태우기 벙커나, 드랍쉽, 커널 등
                       # 331,  # "Move_screen"
                       # 333,  # "Patrol_screen"
                       453,  # "Stop_quick"
                       # 456,  # "Stop_Stop_quick"
                       # 516,  # "UnloadAllAt_screen"
                       ]

        action_spaces = []

        for function_id in intereseted:
            space_size = 1

            for arg in actions.FUNCTIONS[function_id].args:
                if arg.name not in ['queued', 'screen', 'select_point_act']:
                    raise NotImplementedError
                if arg.name == 'screen':
                    space_size *= (env.feature_screen * env.feature_screen)
                else:
                    for size in arg.sizes:
                        space_size *= size

            action_spaces.append({
                'function': actions.FUNCTIONS[function_id],
                'space_size': space_size
            })

    def print_all_actions(self, env, obs):
        for function_id in obs.available_actions:
            print(function_id, env.action_spec[0].functions[function_id].args)


def main():
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('--visualize', type=bool, default=False,
                        help='show the pysc2 visualizer')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='number of episodes to run')
    parser.add_argument('--step-mul', type=int, default=None,
                        help='number of game steps to take per turn')
    parser.add_argument('--random-seed', type=int, default=None,
                        help='the random seed to pass to the game environment')
    args = parser.parse_args()

    example = MoveToBeacon1d(_ENV_NAME, args.visualize, args.step_mul, args.random_seed)
    rewards = example.run(args.num_episodes)
    # if rewards is not None:
    #     print('Total reward: {}'.format(rewards.sum()))
    #     print('Average reward: {} +/- {}'.format(rewards.mean(), rewards.std()))
    #     print('Minimum reward: {}'.format(rewards.min()))
    #     print('Maximum reward: {}'.format(rewards.max()))


if __name__ == "__main__":
    main()
