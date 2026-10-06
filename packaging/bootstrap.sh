#!/usr/bin/env bash
# Provision NEW, isolated workspaces only. Never modifies the historical conda env.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$ROOT"
MAIN_ONLY=0
DRY_RUN=0
EULA=0
RESUME=0
for arg in "$@"; do
    case "$arg" in
        --main-only) MAIN_ONLY=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --accept-eula) EULA=1 ;;
        --resume) RESUME=1 ;;
        --help|-h)
            printf '%s\n' 'Usage: bash packaging/bootstrap.sh [--main-only] [--dry-run] [--resume] --accept-eula' \
                'Linux x86_64, NVIDIA RTX GPU, GLIBC >=2.35. Creates .venv and external/.' \
                'Install uv==0.10.0 and OS build/graphics dependencies first; see root README.md.' \
                'Existing paths are refused unless --resume; resume checks exact Python/source commits.' \
                'Only this repository .venv/external may be modified; historical environments stay untouched.'
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
    esac
done
if (( DRY_RUN )); then
    printf '%s\n' \
        'Main: CPython 3.11.14 -> Isaac Lab 73930be06597075d6e587af1f0a55e0439931f83 + 2 patches' \
        '      -> pip 26.0.1 / requirements.txt / local editable package / dependency verification' \
        'Assets: unitree_ros d96d8f63ae17a7108d4f7229c00ef875ba7129c9'
    if (( ! MAIN_ONLY )); then
        printf '%s\n' 'Arena: CPython 3.12.12 -> c5826c87aa181442328c2c74c49f9e551b5c3d8c + submodules' \
            '       -> uv sync --frozen (separate environment, documented upstream overrides)'
    fi
    exit 0
fi
if (( ! EULA )); then
    printf '%s\n' 'Read NVIDIA licenses first, then pass --accept-eula. No license accepted automatically.' >&2
    exit 2
fi
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    printf '%s\n' 'Only Linux x86_64 was pinned/tested.' >&2; exit 2;
}
command -v git >/dev/null
command -v uv >/dev/null
[[ "$(uv --version)" == 'uv 0.10.0' ]] || {
    printf '%s\n' 'Install the pinned installer: python3 -m pip install --user uv==0.10.0' >&2; exit 2;
}
for target in .venv external/IsaacLab external/unitree_ros; do
    [[ ! -e "$ROOT/$target" || "$RESUME" == 1 ]] || { printf 'Refusing existing path: %s\n' "$ROOT/$target" >&2; exit 2; }
done
if (( ! MAIN_ONLY )); then
    [[ ! -e external/IsaacLab-Arena || "$RESUME" == 1 ]] || { printf '%s\n' 'Arena path already exists; refusing overwrite.' >&2; exit 2; }
fi
# Clean inherited ROS/old Isaac Gym paths before resolving or running either stack.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH LD_PRELOAD
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
# Documentation/test media is not a runtime requirement. WBC fetches its
# licensed model assets separately when first initialized.
export GIT_LFS_SKIP_SMUDGE=1
if [[ ! -e .venv ]]; then
    uv venv --python 3.11.14 --seed .venv
fi
PY="$ROOT/.venv/bin/python"
[[ ! -L .venv ]] || { printf '%s\n' 'Refusing a symlinked .venv.' >&2; exit 2; }
"$PY" -c 'import platform; assert platform.python_version() == "3.11.14"; assert tuple(map(int, platform.libc_ver()[1].split("."))) >= (2,35)'
"$PY" -m pip install pip==26.0.1 setuptools==80.10.2 wheel==0.45.1 toml==0.10.2
clone_pinned() {
    local url="$1" commit="$2" dest="$3"
    if [[ -e "$dest" ]]; then
        [[ ! -L "$dest" && -d "$dest/.git" && "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]] || {
            printf 'Resume source mismatch: %s\n' "$dest" >&2; exit 2;
        }
        return
    fi
    git clone --filter=blob:none --no-checkout "$url" "$dest"
    git -C "$dest" checkout --detach "$commit"
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$commit" ]]
}
mkdir -p external
clone_pinned https://github.com/isaac-sim/IsaacLab.git 73930be06597075d6e587af1f0a55e0439931f83 external/IsaacLab
for patch in isaaclab-current.patch isaaclab-dependency-compat.patch; do
    if ! git -C external/IsaacLab apply --reverse --check "$ROOT/packaging/patches/$patch" 2>/dev/null; then
        git -C external/IsaacLab apply --check "$ROOT/packaging/patches/$patch"
        git -C external/IsaacLab apply "$ROOT/packaging/patches/$patch"
    fi
done
clone_pinned https://github.com/unitreerobotics/unitree_ros.git d96d8f63ae17a7108d4f7229c00ef875ba7129c9 external/unitree_ros
# Resolve all pinned packages in ONE pip operation, not conflicting sequential upgrades.
"$PY" -m pip install -r requirements.txt
"$PY" -m pip check
"$PY" packaging/verify_environment.py --profile main --assets
if (( ! MAIN_ONLY )); then
    clone_pinned https://github.com/isaac-sim/IsaacLab-Arena.git c5826c87aa181442328c2c74c49f9e551b5c3d8c external/IsaacLab-Arena
    # Upstream .gitmodules uses SSH: make public submodules work without keys.
    git -c url.https://github.com/.insteadOf=git@github.com: -C external/IsaacLab-Arena submodule update --init --recursive
    "$PY" -c 'from pathlib import Path; import hashlib; assert hashlib.sha256(Path("external/IsaacLab-Arena/uv.lock").read_bytes()).hexdigest() == "64d912c667eb64776e3c543697cdbcdf16cc97bf7b117d8f2f052128771ac9bf"'
    [[ ! -L external/IsaacLab-Arena/.venv ]] || { printf '%s\n' 'Refusing symlinked Arena .venv.' >&2; exit 2; }
    uv python install 3.12.12
    uv sync --directory external/IsaacLab-Arena --python 3.12.12 --frozen
    external/IsaacLab-Arena/.venv/bin/python packaging/verify_environment.py --profile arena
fi
printf '%s\n' 'Provisioned. Run: source packaging/activate.sh' \
    'Then: unitree-lab doctor --assets --simulation' \
    'For physical teacher/student runs, install the separately distributed datasets/checkpoints/gates first.'
