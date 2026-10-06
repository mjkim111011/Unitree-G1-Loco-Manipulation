"""Packaging checks without simulator launch or modification of existing envs."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]


def test_public_documents_in_sdist_but_local_publish_guide_excluded():
    from setuptools._distutils.filelist import FileList

    research_docs = [
        "G1_43DOF_LOCO_MANIPULATION.md",
        "G1_43DOF_LOCO_MANIPULATION_FRAMEWORK.md",
        "README_G1_43DOF_LOCO_MANIPULATION_DETAILED.md",
        "README_G1_43DOF_PROJECT_STATUS.md",
    ]
    candidates = research_docs + [
        "LICENCE", "README.md", ".python-version", "requirements.txt",
        "requirements/main.snapshot.txt", "requirements/arena.snapshot.txt",
        "packaging/README.md", "packaging/bootstrap.sh", "packaging/vscode.launch.json",
    ]
    assert all((ROOT / value).is_file() for value in candidates)
    files = FileList()
    files.set_allfiles(candidates)
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        files.process_template_line(line)
    assert set(research_docs).issubset(files.files)
    assert "packaging/README.md" not in files.files
    assert "packaging/bootstrap.sh" in files.files
    assert "packaging/vscode.launch.json" in files.files
    assert "requirements/main.snapshot.txt" in files.files
    assert "requirements/arena.snapshot.txt" in files.files


def test_public_readme_does_not_link_excluded_guide():
    assert "packaging/README.md" not in (ROOT / "README.md").read_text()
    assert "packaging/README.md" not in (ROOT / "packaging/bootstrap.sh").read_text()
    assert "/packaging/README.md" in (ROOT / ".gitignore").read_text().splitlines()


def load_paths():
    spec = importlib.util.spec_from_file_location("runtime_paths_test", ROOT / "source/unitree_rl_lab/unitree_rl_lab/runtime_paths.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_root_package_is_installable():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["name"] == "unitree_rl_lab"
    assert data["tool"]["setuptools"]["packages"]["find"]["namespaces"] is True
    assert data["project"]["scripts"]["unitree-lab"] == "unitree_rl_lab.workspace_cli:main"


def test_root_extension_dependency_contract_matches():
    root = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extension = tomllib.loads((ROOT / "source/unitree_rl_lab/pyproject.toml").read_text())["project"]
    assert root["dependencies"] == extension["dependencies"]
    assert root["requires-python"] == extension["requires-python"]
    assert root["version"] == extension["version"]


def test_installer_dry_run_has_no_side_effects():
    result = subprocess.run(["bash", str(ROOT / "packaging/bootstrap.sh"), "--dry-run"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "3.11.14" in result.stdout and "3.12.12" in result.stdout
    assert "uv sync --frozen" in result.stdout


def test_installer_requires_explicit_eula_flag():
    result = subprocess.run(["bash", str(ROOT / "packaging/bootstrap.sh")], capture_output=True, text=True)
    assert result.returncode == 2
    assert "--accept-eula" in result.stderr


def test_installer_main_only_plan():
    result = subprocess.run(["bash", str(ROOT / "packaging/bootstrap.sh"), "--dry-run", "--main-only"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "Arena:" not in result.stdout


def test_portable_launch_is_valid_json():
    text = (ROOT / "packaging/vscode.launch.json").read_text()
    data = json.loads(text)
    assert len(data["configurations"]) == 35
    assert "/home/mjkim" not in text
    for config in data["configurations"]:
        assert config["python"] == "${workspaceFolder}/.venv/bin/python"
        assert config["env"]["UNITREE_ROS_DIR"].endswith("/external/unitree_ros")


def test_requirement_lines_are_fully_pinned():
    for line in (ROOT / "requirements/main.lock.txt").read_text().splitlines():
        if not line or line.startswith(("#", "--")):
            continue
        req = Requirement(line)
        assert req.url or any(s.operator in ("==", "===") for s in req.specifier), line
        if req.url:
            assert len(req.url.rsplit("@", 1)[1].split("#", 1)[0]) == 40


def test_compatibility_changes_are_documented_and_minimal():
    def read_requirements(path):
        result = {}
        for line in path.read_text().splitlines():
            if line and not line.startswith(("#", "--")):
                req = Requirement(line.split(" #", 1)[0])
                if not req.url:
                    result[canonicalize_name(req.name)] = next(iter(req.specifier)).version
        return result
    original = read_requirements(ROOT / "requirements/main.snapshot.txt")
    install = read_requirements(ROOT / "requirements/main.lock.txt")
    differences = {name: version for name, version in install.items() if original[name] != version}
    sources = json.loads((ROOT / "packaging/sources.json").read_text())
    assert differences == sources["compatibility_changes"]
    assert "swanlab" not in install
    assert install["torchaudio"] == "2.7.0"


@pytest.mark.parametrize("variable", ["UNITREE_ARENA_ROOT", "LM_ARENA_ROOT"])
def test_custom_arena_root(monkeypatch, tmp_path, variable):
    for key in ("UNITREE_ARENA_ROOT", "LM_ARENA_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv(variable, str(tmp_path / "worker"))
    assert load_paths().default_arena_root() == str(tmp_path / "worker")


def test_workspace_worker_default(monkeypatch, tmp_path):
    for key in ("UNITREE_ARENA_ROOT", "LM_ARENA_ROOT"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("UNITREE_WORKSPACE", str(tmp_path))
    (tmp_path / "external/IsaacLab-Arena").mkdir(parents=True)
    assert load_paths().default_arena_root() == str(tmp_path / "external/IsaacLab-Arena")


def test_legacy_worker_remains_default_without_activation(monkeypatch, tmp_path):
    for key in ("UNITREE_ARENA_ROOT", "LM_ARENA_ROOT", "UNITREE_WORKSPACE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    (tmp_path / "IsaacLab-Arena").mkdir()
    assert load_paths().default_arena_root() == str(tmp_path / "IsaacLab-Arena")


def test_explicit_dataset_default(monkeypatch):
    monkeypatch.setenv("UNITREE_ARENA_DATASET", "/custom/demo.hdf5")
    assert load_paths().default_arena_dataset() == "/custom/demo.hdf5"


def test_scripts_no_longer_embed_developer_home():
    for filename in ("collect_nominal_box_teacher.py", "collect_carry_place_teacher.py", "collect_dagger_corrections.py",
                     "evaluate_bc2_rollout.py", "play_bc_skill_fsm.py", "generate_randomized_box_teacher.py"):
        text = (ROOT / "scripts/teacher" / filename).read_text()
        assert 'default="/home/mjkim/IsaacLab-Arena' not in text


def test_cli_forwards_subcommand_help(monkeypatch):
    spec = importlib.util.spec_from_file_location("workspace_cli_test", ROOT / "source/unitree_rl_lab/unitree_rl_lab/workspace_cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["unitree-lab", "train-bc", "--help"])
    monkeypatch.setenv("UNITREE_WORKSPACE", str(ROOT))
    calls = []
    monkeypatch.setattr(module.subprocess, "call", lambda command, cwd: calls.append(command) or 0)
    assert module.main() == 0
    assert calls[0][-1] == "--help"
    assert calls[0][-2].endswith("phase_bc.py")


def test_shell_syntax():
    for filename in ("bootstrap.sh", "activate.sh"):
        result = subprocess.run(["bash", "-n", str(ROOT / "packaging" / filename)])
        assert result.returncode == 0
