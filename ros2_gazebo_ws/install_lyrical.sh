#!/usr/bin/env bash
set -eo pipefail
workspace_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$workspace_dir"

if [[ "${ROS_DISTRO:-}" != "lyrical" ]]; then
  echo "Source ROS 2 Lyrical first: source /opt/ros/lyrical/setup.bash" >&2
  exit 1
fi

sudo apt update
sudo apt install -y \
  ros-lyrical-ros-gz \
  ros-lyrical-actuator-msgs \
  python3-colcon-common-extensions \
  python3-matplotlib \
  python3-tk \
  python3-venv \
  python3.14-venv \
  python3-rosdep

# A failed venv creation leaves an incomplete directory without an activate script.
if [[ -d .venv && ! -f .venv/bin/activate ]]; then
  rm -rf -- .venv
fi
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'stable-baselines3>=2.4,<3' numpy

rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

echo "Build complete. Run: source install/setup.bash"
