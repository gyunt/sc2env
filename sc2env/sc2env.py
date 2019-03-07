import logging

import gym
import numpy as np
from gym import spaces
from pysc2.env import sc2_env
from pysc2.env.environment import StepType
from pysc2.lib import actions, features

logger = logging.getLogger(__name__)

_NO_OP = actions.FUNCTIONS.no_op.id
_DEFAULT_SELECTED_FEATURES = [
    'player_relative',
    'selected',
    # 'power',
    # 'unit_energy',
    # 'unit_shields',
    # 'unit_density',
    # 'unit_hit_points',
    # 'visibility_map'
]


class SC2Env(gym.Env):
    metadata = {'render.modes': [None, 'human']}

    def __init__(self,
                 map_name,
                 feature_screen=None,
                 feature_minimap=None,
                 rgb_screen=None,
                 rgb_minimap=None,
                 action_space=None,
                 camera_width_world_units=None,
                 use_feature_units=False,
                 use_raw_units=False,
                 use_unit_counts=False,
                 use_camera_position=False,
                 visualize=False,
                 step_mul=None,
                 random_seed=None,
                 flatten_action=True,

                 selected_features=_DEFAULT_SELECTED_FEATURES, **settings):
        """Creates an AgentInterfaceFormat object from keyword args.

        Convenient when using dictionaries or command-line arguments for config.

        Note that the feature_* and rgb_* properties define the respective spatial
        observation dimensions and accept:
              * None or 0 to disable that spatial observation.
              * A single int for a square observation with that side length.
              * A (int, int) tuple for a rectangular (width, height) observation.

        Args:
            feature_screen: If specified, so must feature_minimap be.
            feature_minimap: If specified, so must feature_screen be.
            rgb_screen: If specified, so must rgb_minimap be.
            rgb_minimap: If specified, so must rgb_screen be.
            action_space: ["FEATURES", "RGB"].
            camera_width_world_units: An int.
            use_feature_units: A boolean, defaults to False.
            use_raw_units: A boolean, defaults to False.
            use_unit_counts: A boolean, defaults to False.
            use_camera_position: A boolean, defaults to False.

        Returns:
            An `AgentInterfaceFormat` object.

        Raises:
            ValueError: If an invalid parameter is specified.
        """

        super().__init__()

        self.feature_screen = feature_screen
        self.feature_minimap = feature_minimap
        self.rgb_screen = rgb_screen
        self.rgb_minimap = rgb_minimap
        self.action_space = action_space
        self.camera_width_world_units = camera_width_world_units
        self.use_feature_units = use_feature_units
        self.use_raw_units = use_raw_units
        self.use_unit_counts = use_unit_counts
        self.use_camera_position = use_camera_position
        self.visualize = visualize
        self.step_mul = step_mul
        self.random_seed = random_seed

        self._selected_features = selected_features
        self._settings = {
            'visualize': self.visualize,
            'step_mul': self.step_mul,
            'random_seed': self.random_seed,
            'map_name': map_name,
            'realtime': False,
            'agent_interface_format': sc2_env.parse_agent_interface_format(
                feature_screen=self.feature_screen,
                feature_minimap=self.feature_minimap,
                rgb_screen=self.rgb_screen,
                rgb_minimap=self.rgb_minimap,
                action_space=self.action_space,
                camera_width_world_units=self.camera_width_world_units,
                use_feature_units=self.use_feature_units,
                use_raw_units=self.use_raw_units,
                use_unit_counts=self.use_unit_counts,
                use_camera_position=self.use_camera_position
            )
        }

        self._env = None
        self.action_info = {}
        self.interested = [
            # 0,  # "no_op"
            # 2,  # "select_point"
            # 3,  # "select_rect"
            # 4,  # "select_control_group"
            # 5,  # "select_unit"
            7,  # "select_army"
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
        self.function_ids = []

        self.flatten_action = flatten_action
        self._episode = 0
        self._num_step = 0
        self._episode_reward = 0
        self._total_reward = 0
        self._max_episode_steps = 200
        self.init_action_space()
        self.init_observation_space()

    def init_action_space(self):
        index = 1
        shape = [len(self.interested)]

        for function_id in self.interested:
            self.function_ids.append(function_id)

            for arg in actions.FUNCTIONS[function_id].args:
                if arg.name == 'queued':
                    continue
                elif arg.name == 'select_add':
                    continue

                if arg.name not in self.action_info:
                    if 'screen' in arg.name:
                        sizes = [self.feature_screen, self.feature_screen]
                    else:
                        sizes = list(arg.sizes)

                    self.action_info[arg.name] = {
                        'size': sizes,
                        'range': range(index, index + len(sizes))
                    }
                    shape += sizes
                    index += len(sizes)

        from pprint import pprint
        pprint(self.action_info)
        print(shape)

        self.action_space = spaces.MultiDiscrete(shape)

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
                                                   len(self._selected_features)),
                                            dtype=np.int16)

    def step(self, action):
        action = self._translate_action(action)

        logger.debug("step: {}".format(action))
        return self._safe_step(action)

    def _translate_action(self, action):
        if not self.flatten_action:
            return action

        function_id = self.function_ids[action[0]]
        args = []

        for arg in self.action_spec[0].functions[function_id].args:
            if arg.name in ['queued', 'select_add']:
                a = [0]
            else:
                a = []
                for i in self.action_info[arg.name]['range']:
                    a.append(action[i])

            if 'screen' in arg.name:
                assert len(arg.sizes) == 2
                a[0], a[1] = a[1], a[0]
            args.append(a)

        this_action = [function_id] + args
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
        obs = np.concatenate(
            [np.expand_dims(feature_screen[feature.name], axis=-1)
             for feature in features.SCREEN_FEATURES
             if feature.name in self._selected_features],
            axis=-1)

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
        print(self.settings)
        self._env = sc2_env.SC2Env(**self.settings)

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

    def seed(self, seed=None):
        self.random_seed = seed

        self._settings.update({
            'random_seed': self.random_seed
        })

    def render(self, mode='human', **kwargs):
        pass
