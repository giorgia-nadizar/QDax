from __future__ import annotations

from functools import partial
from typing import Union, Tuple, Callable

import flax.struct
import jax.lax
import jax.tree_util
from brax.jumpy import cond
from jax import numpy as jnp, vmap

from qdax.environments.kheperax.geoms import Pos, Disk, Segment
from qdax.environments.kheperax.laser import Laser
from qdax.environments.kheperax.maze import Maze
from qdax.environments.kheperax.posture import Posture
from qdax.environments.kheperax.tree_utils import get_batch_size


@flax.struct.dataclass
class Robot:
    posture: Posture
    radius: float
    range_lasers: Union[float, jnp.ndarray]
    laser_angles: jnp.ndarray
    std_noise_sensor_measures: float = flax.struct.field(pytree_node=False, default=0.)

    # Makes the controllers more difficult to learn
    lasers_return_minus_one_if_out_of_range: bool = flax.struct.field(pytree_node=False, default=False)

    @classmethod
    def create_default_robot(cls):
        return cls(
            posture=Posture(0.15, 0.15, jnp.pi / 2),
            radius=0.015,
            range_lasers=0.2,
            laser_angles=jnp.array([-jnp.pi / 4, 0, jnp.pi / 4]),
            std_noise_sensor_measures=0.,
            lasers_return_minus_one_if_out_of_range=False,
        )

    def get_lasers(self) -> Laser:
        list_lasers = []

        for laser_angle in self.laser_angles:
            _laser = Laser(Pos(self.posture.x, self.posture.y), self.posture.angle + laser_angle, self.range_lasers)
            list_lasers.append(_laser)

        tree_lasers = jax.tree_util.tree_map(lambda *x: jnp.asarray(x, dtype=jnp.float32),
                                             *list_lasers)
        return tree_lasers

    def get_disk(self) -> Disk:
        return Disk(Pos(self.posture.x, self.posture.y),
                    self.radius)

    def collides(self, maze: Maze) -> bool:
        return self.get_disk().collides(maze.walls)

    def move(self, v1, v2, maze: Maze) -> Tuple[Robot, jnp.ndarray]:
        previous_robot = self
        new_posture = self.posture.move(v1, v2, 2 * self.radius)

        new_robot = self.replace(posture=new_posture)

        def if_collides(_):
            return previous_robot

        def if_not_collides(_):
            return new_robot

        next_robot = jax.lax.cond(self.collides(maze), if_collides, if_not_collides, None)

        return next_robot, self.bumper_measures(maze)

    def laser_measures(self, maze: Maze, random_key) -> jnp.ndarray:
        lasers = self.get_lasers()
        number_lasers = get_batch_size(lasers)

        random_key, *subkeys = jax.random.split(random_key, num=number_lasers + 1)
        array_subkeys = jnp.asarray(subkeys)
        get_measure_v = jax.vmap(lambda laser, _subkey: laser.get_measure(array_segments=maze.walls,
                                                                          return_minus_one_if_out_of_range=self.lasers_return_minus_one_if_out_of_range,
                                                                          std_noise=self.std_noise_sensor_measures,
                                                                          random_key=_subkey))
        return get_measure_v(self.get_lasers(), array_subkeys)

    def bumper_measure_with_segment(self, segment: Segment) -> jnp.ndarray:
        pos = Pos.from_posture(self.posture)
        projection = pos.calculate_projection_on_segment(segment)
        vector = projection - pos
        angle = jnp.arctan2(vector.y, vector.x)
        angle_diff = jnp.mod(angle - self.posture.angle + jnp.pi, 2 * jnp.pi) - jnp.pi

        distance = projection.dist_to(pos)

        left_bumper_collides = jnp.logical_and(
            jnp.logical_and(angle_diff <= 0., angle_diff > -jnp.pi / 2),
            distance < self.radius
        )
        right_bumper_collides = jnp.logical_and(
            jnp.logical_and(angle_diff >= 0., angle_diff < jnp.pi / 2),
            distance < self.radius
        )

        left_bumper_measure = jnp.where(left_bumper_collides, 1., -1.)
        right_bumper_measure = jnp.where(right_bumper_collides, 1., -1.)

        return jnp.array([left_bumper_measure, right_bumper_measure])

    def bumper_measures(self, maze: Maze) -> jnp.ndarray:
        get_measure_v = jax.vmap(lambda segment: self.bumper_measure_with_segment(segment))
        return jnp.max(get_measure_v(maze.walls), axis=0)


@flax.struct.dataclass
class SightEvolvableRobot(Robot):
    desired_laser_angles: jnp.ndarray = jnp.array([])

    @staticmethod
    def _interpolate(y_l: float, y_r: float, x_l: float, x_r: float, x: float) -> float:
        return (x - x_r) * (y_l - y_r) / (x_l - x_r) + y_r

    @staticmethod
    def _single_angle_interpolation(
            target_angle: float,
            sight_readings: jnp.ndarray,
            real_angles: jnp.ndarray) -> float:
        idx_l = jnp.searchsorted(real_angles, target_angle, side='right')
        idx_r = jnp.searchsorted(real_angles, target_angle, side='right') + 1
        real_angles_processed = jnp.concatenate([
            jnp.asarray([real_angles[len(real_angles) - 1] - 2 * jnp.pi]),
            real_angles,
            jnp.asarray([real_angles[0] + 2 * jnp.pi])
        ])
        sight_readings_processed = jnp.concatenate([
            jnp.asarray([sight_readings[len(sight_readings) - 1]]),
            sight_readings,
            jnp.asarray([sight_readings[0]])
        ])
        x_l, x_r = real_angles_processed[idx_l], real_angles_processed[idx_r]
        y_l, y_r = sight_readings_processed[idx_l], sight_readings_processed[idx_r]
        return SightEvolvableRobot._interpolate(y_l, y_r, x_l, x_r, target_angle)

    @classmethod
    def from_robot_and_genotype(cls, robot: Robot,
                                genotype: jnp.ndarray,
                                desired_laser_angles: jnp.ndarray) -> SightEvolvableRobot:
        real_laser_angles = jnp.tanh(genotype) * jnp.pi  # defined in [-pi, pi]
        return SightEvolvableRobot.from_robot(robot, real_laser_angles, desired_laser_angles)

    @classmethod
    def from_robot(cls,
                   robot: Robot,
                   real_laser_angles: jnp.ndarray,
                   desired_laser_angles: jnp.ndarray
                   ) -> SightEvolvableRobot:
        desired_laser_angles = jnp.sort(desired_laser_angles)
        real_laser_angles = jnp.sort(real_laser_angles)
        return cls(
            posture=robot.posture,
            radius=robot.radius,
            range_lasers=robot.range_lasers,
            laser_angles=real_laser_angles,
            std_noise_sensor_measures=robot.std_noise_sensor_measures,
            lasers_return_minus_one_if_out_of_range=robot.lasers_return_minus_one_if_out_of_range,
            desired_laser_angles=desired_laser_angles,
        )

    # the values returned are obtained from the actual ones via interpolation, as if perceived from the desired ones
    def laser_measures(self, maze: Maze, random_key) -> jnp.ndarray:
        real_measures = super().laser_measures(maze, random_key)
        single_angle_interpolation_func = partial(SightEvolvableRobot._single_angle_interpolation,
                                                  sight_readings=real_measures,
                                                  real_angles=self.laser_angles,
                                                  )
        return vmap(single_angle_interpolation_func)(self.desired_laser_angles)
