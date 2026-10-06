"""Finite Isaac Sim smoke test, without checkpoints, datasets, or Arena."""
import os
import faulthandler
from pathlib import Path

# Avoid shared /tmp/isaaclab/logs left by a different user/session.
log_temp = Path(__file__).resolve().parent / "validation" / "tmp"
log_temp.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(log_temp)
faulthandler.dump_traceback_later(60, repeat=True)

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app
print("ISAAC_APP_STARTED", flush=True)
simulation = None
try:
    import isaaclab.sim as sim_utils

    simulation = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    print("ISAAC_CONTEXT_CREATED", flush=True)
    simulation.reset()
    print("ISAAC_CONTEXT_RESET", flush=True)
    for _ in range(10):
        simulation.step()
    print("ISAAC_SIM_SMOKE_OK", flush=True)
finally:
    # Unsubscribe Isaac Lab's timeline-stop handler before Kit closes the
    # stage; otherwise its standalone paused-render loop blocks shutdown.
    if simulation is not None:
        simulation.clear_instance()
    app.close()
    faulthandler.cancel_dump_traceback_later()
