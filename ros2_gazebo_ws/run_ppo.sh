#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"
source /opt/ros/lyrical/setup.bash
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
if [[ ! -f install/setup.bash ]]; then
  echo "ROS workspace is not built: $workspace_dir/install/setup.bash is missing." >&2
  echo "Run ./install_lyrical.sh and make sure 'colcon build' completes successfully." >&2
  exit 1
fi
source install/setup.bash

venv_python="$workspace_dir/.venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "The workspace virtual environment is missing: $venv_python" >&2
  echo "Run ./install_lyrical.sh first." >&2
  exit 1
fi
if ! "$venv_python" -c 'import stable_baselines3, torch' >/dev/null 2>&1; then
  echo "PPO dependencies are not installed in .venv." >&2
  echo "Run: .venv/bin/python -m pip install 'stable-baselines3>=2.4,<3' torch" >&2
  exit 1
fi

# Colcon may generate the ROS console script with /usr/bin/python3. Make the
# venv packages visible to that interpreter while retaining ROS system packages.
venv_site_packages="$("$venv_python" -c 'import site; print(site.getsitepackages()[0])')"
export PYTHONPATH="$venv_site_packages${PYTHONPATH:+:$PYTHONPATH}"

checkpoint="${1:-}"
if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
  echo "Usage: $0 /absolute/path/to/ppo_6dof_seed_9.zip" >&2
  exit 2
fi
echo "Gazebo will start PAUSED while PyTorch and the PPO checkpoint load."
echo "Wait for 'controller=ppo' below, then click Gazebo's Play button."
ros2 launch quadrotor_gz sim.launch.py \
  controller:=ppo checkpoint:="$checkpoint" start_paused:=true
