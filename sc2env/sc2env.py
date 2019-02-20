import gym
import numpy as np
from gym import spaces
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features

_NO_OP = actions.FUNCTIONS.no_op.id
_DEFAULT_SELECTED_FEATURES = [
                                  # 'player_relative',
                                  'selected',
                                  'power',
                                  # 'unit_energy',
                                  # 'unit_shields',
                                  # 'unit_density',
                                  # 'unit_hit_points',
                                  # 'visibility_map'
                              ]


class SC2Env(gym.Env):
    metadata = {'render.modes': [None, 'human']}
    default_settings = {'agent_interface_format': sc2_env.parse_agent_interface_format(
        feature_screen=84,
        feature_minimap=64,
    )}

    def __init__(self,
                 flatten_action=True,
                 feature_screen=84,
                 feature_minimap=64,
                 selected_features=_DEFAULT_SELECTED_FEATURES, **settings) -> None:
        super().__init__()

        self.feature_screen = feature_screen
        self.feature_minimap = feature_minimap
        self._selected_features = selected_features

        self._settings = {
            'visualize': False,
            'step_mul': None,
            'random_seed': None,
            'map_name': 'MoveToBeacon',

            'agent_interface_format': sc2_env.parse_agent_interface_format(
                feature_screen=feature_screen,
                feature_minimap=feature_minimap,
            )
        }

        self._env = None
        self.action_space_info = None

        self.flatten_action = flatten_action
        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0
        self.init_action_space()
        self.init_observation_space()

    def init_action_space(self):
        intereseted = [#0,  # "no_op"
                       2,  # "select_point"
                       # 3,  # "select_rect"
                       # 4,  # "select_control_group"
                       # 5,  # "select_unit"
                       # 12,  # "attack_screen"
                       # 14,  # "attack_attack_screen"
                       # 140,  # "Cancel_quick"
                       # 261,  # "Halt_quick"
                       # 264,  # "Harvest_Gather_screen"
                       # 269,  # "Harvest_Return_quick"
                       # 274,  # "HoldPosition_quick"
                       # 275,  # "Land_screen"
                       # 281,  # "Lift_quick"
                       # 287,  # "Load_screen" 태우기 벙커나, 드랍쉽, 커널 등
                       331,  # "Move_screen"
                       # 333,  # "Patrol_screen"
                       # 453,  # "Stop_quick"
                       # 456,  # "Stop_Stop_quick"
                       # 516,  # "UnloadAllAt_screen"
                       ]

        action_spaces = []
        total_space_size = 0

        for function_id in intereseted:
            space_size = 1

            for arg in actions.FUNCTIONS[function_id].args:
                if arg.name not in ['queued', 'screen', 'select_point_act']:
                    raise NotImplementedError
                if arg.name == 'screen':
                    space_size *= (self.feature_screen * self.feature_screen)
                else:
                    for size in arg.sizes:
                        space_size *= size

            action_spaces.append({
                'id': function_id,
                'function': actions.FUNCTIONS[function_id],
                'space_size': space_size
            })
            total_space_size += space_size

        self.action_space_info = action_spaces
        self.action_space = spaces.Discrete(total_space_size)

    def init_observation_space(self):
        # see pysc2.lib.features.observation_spec
        # see https://docs.rs/crate/sc2-proto/

        # feature_screen = {}
        # for feature in features.SCREEN_FEATURES:
        #     if feature.name in self._selected_features:
        #         feature_screen[feature.name] = spaces.Box(low=0, high=feature.scale, shape=(
        #             self.feature_screen, self.feature_screen))

        # TODO (Gyunt) Currently Use Only feature_screen.
        # self.observation_space = spaces.Dict(
        #     {
        #         # "action_result": spaces.Discrete(214), # See error.proto: ActionResult.
        #         # "alerts": (0,),  # See sc2api.proto: Alert.
        #         # "available_actions": (0,),
        #         # "build_queue": spaces.Discrete(len(features.UnitLayer)),  # pytype: disable=wrong-arg-types
        #         # "cargo": (0, len(UnitLayer)),  # pytype: disable=wrong-arg-types
        #         # "cargo_slots_available": (1,),
        #         # "control_groups": (10, 2),
        #         # "game_loop": (1,),
        #         # "last_actions": (0,),
        #         # "multi_select": (0, len(UnitLayer)),  # pytype: disable=wrong-arg-types
        #         # "player": spaces.Discrete(len(features.Player)),  # pytype: disable=wrong-arg-types
        #         # "score_cumulative": spaces.Discrete(len(features.ScoreCumulative)),  # pytype: disable=wrong-arg-types
        #         # "score_by_category": (len(ScoreByCategory), len(ScoreCategories)),  # pytype: disable=wrong-arg-types
        #         # "score_by_vital": (len(ScoreByVital), len(ScoreVitals)),  # pytype: disable=wrong-arg-types
        #         # "single_select": spaces.Discrete(len(features.UnitLayer)),  # Only (n, 7) for n in (0, 1).  # pytype: disable=wrong-arg-types
        #         # "multi_select"
        #         # "feature_screen": spaces.Dict(feature_screen)
        #     }
        # )
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(self.feature_screen,
                                                   self.feature_screen,
                                                   len(self._selected_features)))

    def step(self, action):
        action = self._translate_action(action)
        return self._safe_step(action)

    def _translate_action(self, action):
        if not self.flatten_action:
            return action

        index = action
        for info in self.action_space_info:
            if index < info['space_size']:
                function = info['function']
                function_id = info['id']
                break
            index -= info['space_size']

        args = []
        space_size = 1
        for arg in self.action_spec[0].functions[function_id].args:
            for size in arg.sizes:
                space_size *= size

        for arg in self.action_spec[0].functions[function_id].args:
            a = []
            for size in arg.sizes:
                space_size //= size

                a.append(index // space_size)
                index %= space_size
            args.append(a)

        this_action = [function_id] + args
        # print(this_action)
        return this_action

    def _safe_step(self, action):
        self._num_step += 1
        if action[0] not in self.available_actions:
            action = [_NO_OP]
        try:
            obs = self._env.step([actions.FunctionCall(action[0], action[1:])])[0]
        except Exception:
            return None, 0, True, {}
        self.available_actions = obs.observation['available_actions']
        reward = obs.reward
        self._episode_reward += reward
        self._total_reward += reward
        return self._reduce_observation(obs.observation), reward, obs.step_type == StepType.LAST, {}

    def _reduce_observation(self, observation_from_api):
        # obs = {}
        feature_screen = observation_from_api['feature_screen']
        # for feature in features.SCREEN_FEATURES:
        #     if feature.name in self._selected_features:
        #         obs.update({feature.name: feature_screen[feature.name]})
        # tmp = [np.expand_dims(feature_screen[feature.name]) for feature in features.SCREEN_FEATURES if
        #                       feature.name in self._selected_features]
        obs = np.concatenate([np.expand_dims(feature_screen[feature.name], axis=-1) for feature in features.SCREEN_FEATURES if
                              feature.name in self._selected_features], axis=-1)
        return obs

    def reset(self):
        if self._env is None:
            self._init_env()
        self._episode += 1
        self._num_step = 0
        self._episode_reward = 0
        obs = self._env.reset()[0]
        self.available_actions = obs.observation['available_actions']
        return self._reduce_observation(obs.observation)

    def save_replay(self, replay_dir):
        self._env.save_replay(replay_dir)

    def _init_env(self):
        args = {**self.default_settings, **self._settings}
        print(args)
        self._env = sc2_env.SC2Env(**args)

    def close(self):
        if self._env is not None:
            self._env.close()
        super().close()

    @property
    def settings(self):
        return self._settings

    @property
    def action_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.action_spec()

    @property
    def observation_spec(self):
        if self._env is None:
            self._init_env()
        return self._env.observation_spec()

    @property
    def episode(self):
        return self._episode

    @property
    def num_step(self):
        return self._num_step

    @property
    def episode_reward(self):
        return self._episode_reward

    @property
    def total_reward(self):
        return self._total_reward
