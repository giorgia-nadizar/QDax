from __future__ import annotations

import dataclasses
from typing import List, Tuple, Dict, Any, Union

import brax.envs
import flax.struct
import jax.tree_util
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.bd_extractors import (
    get_final_xy_position,
)

from qdax.tasks.brax_envs import create_brax_scoring_fn
from qdax.types import RNGKey

from qdax.environments.kheperax.maze import Maze
from qdax.environments.kheperax.rendering_tools import RenderingTools
from qdax.environments.kheperax.robot import Robot
from qdax.environments.kheperax.type_fixer_wrapper import TypeFixerWrapper


@dataclasses.dataclass
class KheperaxConfig:
    episode_length: int
    mlp_policy_hidden_layer_sizes: Tuple[int, ...]
    action_scale: float
    maze: Maze
    robot: Robot
    std_noise_wheel_velocities: float
    resolution: Tuple[int, int]

    @classmethod
    def with_episode_length(cls, episode_length: int):
        return cls(
            episode_length=episode_length,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=(64, 64),
            action_scale=0.025,
            maze=Maze.create_default_maze(),
            robot=Robot.create_default_robot(),
            std_noise_wheel_velocities=0.0,
        )

    @classmethod
    def get_default(cls):
        return cls(
            episode_length=250,
            mlp_policy_hidden_layer_sizes=(8,),
            resolution=(64, 64),
            action_scale=0.025,
            maze=Maze.create_default_maze(),
            robot=Robot.create_default_robot(),
            std_noise_wheel_velocities=0.0,
        )


@flax.struct.dataclass
class KheperaxState:
    """
    Environment state for training and inference.
    """
    robot: Robot
    maze: Maze
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    random_key: RNGKey
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, Any] = flax.struct.field(default_factory=dict)


class KheperaxTask(brax.envs.env.Env):
    def __init__(self, kheperax_config: KheperaxConfig, **kwargs):
        self.kheperax_config = kheperax_config
        super().__init__(None, **kwargs)

    @classmethod
    def create_environment(cls, kheperax_config: KheperaxConfig):
        env = cls(kheperax_config)
        env = brax.envs.wrappers.EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=1)
        env = TypeFixerWrapper(env)
        return env

    @classmethod
    def create_default_task(cls,
                            kheperax_config: KheperaxConfig,
                            random_key,
                            ):
        env = cls.create_environment(kheperax_config)
        bd_extraction_fn = get_final_xy_position

        # Init policy network
        policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (env.action_size,)
        policy_network = MLP(
            layer_sizes=policy_layer_sizes,
            kernel_init=jax.nn.initializers.lecun_uniform(),
            final_activation=jnp.tanh,
        )

        scoring_fn, random_key = create_brax_scoring_fn(
            env,
            policy_network,
            bd_extraction_fn,
            random_key,
            episode_length=kheperax_config.episode_length,
        )

        return env, policy_network, scoring_fn

    def get_xy_pos(self, robot: Robot):
        return jnp.asarray([robot.posture.x, robot.posture.y])

    def reset(self, random_key: jnp.ndarray) -> KheperaxState:
        robot = self.kheperax_config.robot

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(robot, self.kheperax_config.maze, random_key=subkey)
        reward = 0.
        done = False

        info = {
            "state_descriptor": self.get_xy_pos(robot),
        }

        random_key, subkey = jax.random.split(random_key)

        return KheperaxState(
            maze=self.kheperax_config.maze,
            robot=robot,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            random_key=subkey,
        )

    def _get_wheel_velocities(self, action, random_key):
        random_key, subkey = jax.random.split(random_key)
        noise_wheel_velocities = jax.random.normal(subkey, shape=action.shape) \
                                 * self.kheperax_config.std_noise_wheel_velocities

        scale_actions = self.kheperax_config.action_scale
        wheel_velocities = action * scale_actions
        wheel_velocities = wheel_velocities + noise_wheel_velocities
        wheel_velocities = jnp.clip(wheel_velocities, -scale_actions, scale_actions)

        return wheel_velocities

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)

        # reward penalizes high action values
        reward = -1. * jnp.power(jnp.linalg.norm(wheel_velocities), 2.)

        # only stop at the end of the episode
        done = False

        state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )

    def _get_obs(self, robot: Robot, maze: Maze, random_key, bumper_measures: jnp.ndarray = None) -> jnp.ndarray:
        random_key, subkey = jax.random.split(random_key)
        laser_measures = robot.laser_measures(maze, random_key=subkey)

        if bumper_measures is None:
            bumper_measures = robot.bumper_measures(maze)

        return jnp.concatenate([laser_measures, bumper_measures])

    @property
    def observation_size(self) -> int:
        number_bumpers = 2
        number_lasers = len(self.kheperax_config.robot.laser_angles)
        return number_bumpers + number_lasers

    @property
    def action_size(self) -> int:
        return 2

    @property
    def state_descriptor_length(self) -> int:
        return 2

    @property
    def behavior_descriptor_limits(self) -> Tuple[List[float], List[float]]:
        return [0., 0.], [1., 1.]

    @property
    def backend(self) -> str:
        return "Kheperax"

    def render(self, state: KheperaxState, ) -> jnp.ndarray:
        # WARNING: only consider the maze is in the unit square
        coeff_triangle = 3.
        image = jnp.zeros(self.kheperax_config.resolution, dtype=jnp.float32)
        image = RenderingTools.place_triangle(image,
                                              point_1=(
                                                  state.robot.posture.x + coeff_triangle * state.robot.radius * jnp.cos(
                                                      state.robot.posture.angle),
                                                  state.robot.posture.y + coeff_triangle * state.robot.radius * jnp.sin(
                                                      state.robot.posture.angle)),
                                              point_2=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle - jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle - jnp.pi / 2)),
                                              point_3=(state.robot.posture.x + state.robot.radius * jnp.cos(
                                                  state.robot.posture.angle + jnp.pi / 2),
                                                       state.robot.posture.y + state.robot.radius * jnp.sin(
                                                           state.robot.posture.angle + jnp.pi / 2)),
                                              value=2.)

        image = RenderingTools.place_circle(image,
                                            center=(state.robot.posture.x, state.robot.posture.y),
                                            radius=state.robot.radius,
                                            value=1.)

        white = jnp.array([1., 1., 1.])
        blue = jnp.array([0., 0., 1.])
        red = jnp.array([1., 0., 0.])
        green = jnp.array([0., 1., 0.])
        magenta = jnp.array([0.5, 0., 1.])
        cyan = jnp.array([0., 1., 1.])
        yellow = jnp.array([1., 1., 0.])
        black = jnp.array([0., 0., 0.])

        image = RenderingTools.place_segments(image,
                                              state.maze.walls,
                                              value=5.)

        index_to_color = {
            0.: white,
            1.: blue,
            2.: magenta,
            5.: black,
        }

        def _map_colors(x):
            new_array = jnp.zeros((3,))
            for key, value in index_to_color.items():
                new_array = jax.lax.cond(jnp.isclose(x, key), lambda _: value, lambda _: new_array, operand=None)
            return new_array

        image = jax.vmap(jax.vmap(_map_colors))(image)

        return image


class KheperaxTargetedTask(KheperaxTask):
    target_point: jnp.ndarray

    def __init__(self, kheperax_config: KheperaxConfig, target_point: jnp.ndarray, **kwargs):
        self.target_point = target_point
        super().__init__(kheperax_config=kheperax_config, **kwargs)

    @classmethod
    def create_environment(cls, kheperax_config: KheperaxConfig, target_point: jnp.ndarray = jnp.asarray([0.125, 0.9])):
        env = cls(kheperax_config, target_point)
        env = brax.envs.wrappers.EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=1)
        env = TypeFixerWrapper(env)
        return env

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        previous_position = self.get_xy_pos(state.robot)

        random_key = state.random_key

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)
        new_position = state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        # reward describes negative distance from target
        previous_distance = jnp.linalg.norm(self.target_point - previous_position)
        current_distance = jnp.linalg.norm(self.target_point - new_position)
        reward = previous_distance - current_distance

        # only stop at the end of the episode
        done = False

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(  # type: ignore
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )


class KheperaxSpeedTargetedTask(KheperaxTargetedTask):
    target_width: float

    def __init__(self, kheperax_config: KheperaxConfig, target_point: jnp.ndarray, target_width: float, **kwargs):
        self.target_width = target_width
        super().__init__(kheperax_config=kheperax_config, target_point=target_point, **kwargs)

    @classmethod
    def create_environment(
            cls,
            kheperax_config: KheperaxConfig,
            target_point: jnp.ndarray = jnp.asarray([0.125, 0.9]),
            target_width: float = 0.1
    ):
        env = cls(kheperax_config, target_point, target_width)
        env = brax.envs.wrappers.EpisodeWrapper(env, kheperax_config.episode_length, action_repeat=1)
        env = TypeFixerWrapper(env)
        return env

    def _in_zone(self, x_pos: jnp.ndarray, y_pos: jnp.ndarray) -> Union[bool, jnp.ndarray]:
        target_x, target_y = self.target_point.at[0].get(), self.target_point.at[1].get()

        condition_1 = target_x - self.target_width / 2 <= x_pos
        condition_2 = x_pos <= target_x + self.target_width / 2

        condition_3 = target_y - self.target_width / 2 <= y_pos
        condition_4 = y_pos <= target_y + self.target_width / 2

        return condition_1 & condition_2 & condition_3 & condition_4

    def step(self, state: KheperaxState, action: jnp.ndarray) -> KheperaxState:
        random_key = state.random_key

        # determine if zone was reached
        old_xy_pos = self.get_xy_pos(state.robot)
        in_zone = self._in_zone(old_xy_pos[0], old_xy_pos[1])

        done = jnp.where(
            jnp.array(in_zone),
            x=jnp.array(1.0),
            y=jnp.array(0.0),
        )

        # actions should be between -1 and 1
        action = jnp.clip(action, -1., 1.)

        random_key, subkey = jax.random.split(random_key)
        wheel_velocities = self._get_wheel_velocities(action, subkey)

        new_robot, bumper_measures = state.robot.move(wheel_velocities[0], wheel_velocities[1], state.maze)

        random_key, subkey = jax.random.split(random_key)
        obs = self._get_obs(new_robot, state.maze, bumper_measures=bumper_measures, random_key=subkey)
        xy_pos = state.info["state_descriptor"] = self.get_xy_pos(new_robot)

        # reward describes negative distance from target
        reward = -jnp.linalg.norm(xy_pos - self.target_point)

        random_key, subkey = jax.random.split(random_key)
        new_random_key = subkey

        return state.replace(  # type: ignore
            maze=state.maze,
            robot=new_robot,
            obs=obs,
            reward=reward,
            done=done,
            random_key=new_random_key,
        )

    def reset(self, random_key: jnp.ndarray) -> KheperaxState:
        reset_state = super().reset(random_key)
        return reset_state.replace(done=jnp.array(0.0))  # type : ignore
