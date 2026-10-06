# Source this file; do not execute it. Keeps ROS / older simulator libraries out.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf '%s\n' 'Use: source packaging/activate.sh' >&2
    exit 2
fi
export UNITREE_WORKSPACE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$UNITREE_WORKSPACE/.venv/bin/activate" ]]; then
    printf '%s\n' 'Missing .venv; run packaging/bootstrap.sh first.' >&2
    return 2
fi
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
source "$UNITREE_WORKSPACE/.venv/bin/activate"
export UNITREE_ARENA_ROOT="$UNITREE_WORKSPACE/external/IsaacLab-Arena"
export UNITREE_ROS_DIR="$UNITREE_WORKSPACE/external/unitree_ros"
export UNITREE_ARENA_DATASET="$UNITREE_ARENA_ROOT/data/arena_g1_loco_manipulation_dataset_generated_small.hdf5"
export LM_PY="$UNITREE_WORKSPACE/.venv/bin/python" LM_ARENA_ROOT="$UNITREE_ARENA_ROOT"
export LM_ARENA_DATASET="$UNITREE_ARENA_DATASET" PYTHONUNBUFFERED=1
# EULA flags are NOT set here; first simulator start prompts if not already accepted.
