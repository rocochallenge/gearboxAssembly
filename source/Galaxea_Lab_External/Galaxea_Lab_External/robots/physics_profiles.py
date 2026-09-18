"""Opt-in solver budget for faster R1Pro and R1 Lite simulation."""


def use_fast_physics(cfg):
    """Reduce robot solver iterations without changing the timestep or sensors.

    The default robot assets retain their original 128/128 iteration budget.
    Copy before editing so other environments and subsequent configs retain
    their own settings. This profile changes numerical contact resolution;
    seating must be evaluated independently for each robot and profile.
    """
    if cfg.robot_bundle.name not in ("r1_pro", "r1_lite"):
        raise ValueError("--fast_physics is supported only for the r1_pro and r1_lite robot bundles")

    cfg.robot_cfg = cfg.robot_cfg.copy()
    for props in (cfg.robot_cfg.spawn.rigid_props, cfg.robot_cfg.spawn.articulation_props):
        props.solver_position_iteration_count = 32
        props.solver_velocity_iteration_count = 8
