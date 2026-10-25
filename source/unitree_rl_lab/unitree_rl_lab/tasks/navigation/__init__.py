"""Goal-conditioned PPO navigation (Homie [vx, vy, yaw_rate]) to the BC-1 v2 handoff state."""

import gymnasium as gym

gym.register(
    id="Unitree-G1-43dof-BoxNav-Homie",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BoxNavHomiePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-43dof-BoxNav-HomieV6",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV6EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV6EvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BoxNavHomiePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-43dof-BoxNav-HomieV6b",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV6bEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV6bEvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BoxNavHomiePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-43dof-BoxNav-HomieV7",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV7EnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.nav_env_cfg:BoxNavHomieV7EvalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:BoxNavHomiePPORunnerCfg",
    },
)
