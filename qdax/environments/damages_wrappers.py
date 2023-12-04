from brax import jumpy as jp
from brax.envs import Wrapper, State, Env

from qdax.types import Mask


class BrokenSensorsWrapper(Wrapper):

    def __init__(self, env: Env, breakage_mask: Mask):
        assert env.observation_size == len(breakage_mask)
        super().__init__(env)
        self.breakage_mask = breakage_mask

    def step(self, state: State, action: jp.ndarray) -> State:
        state = super().step(state, action)
        observation = state.obs
        masked_observation = self.breakage_mask * observation
        return state.replace(obs=masked_observation)


class BrokenActuatorsWrapper(Wrapper):

    def __init__(self, env: Env, breakage_mask: Mask):
        assert env.action_size == len(breakage_mask)
        super().__init__(env)
        self.breakage_mask = breakage_mask

    def step(self, state: State, action: jp.ndarray) -> State:
        masked_action = self.breakage_mask * action
        state = super().step(state, masked_action)
        return state
