from gymnasium.envs.registration import register

from .kessler_env import KesslerEnv, get_obs

__all__ = [KesslerEnv, get_obs]

register(
    id="envs/KesslerEnv-v0",
    entry_point="envs:KesslerEnv"
)
