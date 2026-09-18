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
ros2 launch quadrotor_gz sim.launch.py controller:=pid
