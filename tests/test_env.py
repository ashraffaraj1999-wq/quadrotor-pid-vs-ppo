from gymnasium.utils.env_checker import check_env
from src.env import Quadrotor6DoFEnv
def test_api(): check_env(Quadrotor6DoFEnv(max_steps=5),skip_render_check=True)

