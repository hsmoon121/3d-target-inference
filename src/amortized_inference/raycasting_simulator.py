import os
import numpy as np
from stable_baselines3 import PPO as PPO_sb3
from tqdm import tqdm
import mujoco

from src import Simulator
from src.rl.sb3.PPO import PPO
from src.utils.functions import natural_sort, parse_yaml, write_yaml


class RaycastingSimulator(object):
  """
  A simulator of target selection behavior with raycasting
  """
  def __init__(self, sim_config=None):
    if sim_config is None:
      config_path="./configs/train.yaml"
      self.dense_grid = True
      self.small_target = -1
    else:
      config_path = sim_config["config_path"]
      self.dense_grid = sim_config["dense_grid"]
      self.small_target = sim_config["small_target"]

    config = parse_yaml(config_path)
    self.shoulder_fixed = config["simulation"]["task"]["kwargs"]["fix_shoulder"]
    
    model_name = "click_dense" if self.dense_grid else "click_wide"
    config["model_name"] = "fixed_" + model_name if self.shoulder_fixed else model_name
    config["simulation"]["task"]["cls"] = "Raycasting"

    if self.small_target == 1:
      config["simulation"]["task"]["kwargs"]["level"] = 2 if self.dense_grid else 4
    elif self.small_target == 0:
      config["simulation"]["task"]["kwargs"]["level"] = 1 if self.dense_grid else 3
    else:
      config["simulation"]["task"]["kwargs"]["level"] = [1, 2] if self.dense_grid else [3, 4]
        
    new_config_path = "./configs/sim.yaml"
    write_yaml(config, new_config_path)
    self._simulator_path = Simulator.build(new_config_path)
    
    self._model_path = os.path.join(self._simulator_path, "../../data/simulator_models", config["model_name"])
    assert len(os.listdir(self._model_path)) > 0, "There is no trained model for simulation. Please train the simulator model first."
    self._model_file = natural_sort(os.listdir(self._model_path))[-1]

    if self.shoulder_fixed:
      pointing_name = "fixed_pointing"
    elif self.dense_grid:
      pointing_name = "pointing_dense"
    else:
      pointing_name = "pointing_wide"
    pointing_ckpt_dir = os.path.join(self._simulator_path, "../../data/simulator_models", pointing_name)
    pointing_model_file = natural_sort(os.listdir(pointing_ckpt_dir))[-1]

    self.action_sample_freq = 20
    run_params = dict()
    run_params["action_sample_freq"] = self.action_sample_freq
    run_params["evaluate"] = True
    self._env = Simulator.get(self._simulator_path, run_parameters=run_params)

    _model = PPO(self._env, model_name=config["model_name"])
    custom_objects = {"policy_kwargs": _model.rl_config["policy_kwargs"]}
    print(f"Loading [ click ] model: {os.path.join(self._model_path, self._model_file)}")
    self.model = PPO_sb3.load(os.path.join(self._model_path, self._model_file), custom_objects=custom_objects)
    print(f"Loading [ pointing ] model: {os.path.join(pointing_ckpt_dir, pointing_model_file)}")
    self.pointing_model = PPO_sb3.load(os.path.join(pointing_ckpt_dir, pointing_model_file), custom_objects=custom_objects)

    assert self._env.bm_model._with_user_params
    self.norm_info = dict(
      max_target_pos = np.array([5.0, 1.6, 1.6]), # (depth, x, y) orders
      min_target_pos = np.array([4.5, -1.6, -1.6]),
      max_user_params = self._env.bm_model.user_params_max,
      min_user_params = self._env.bm_model.user_params_min,
      max_summary = np.array([0.05] + [5.0, 1.6, 1.6] * len(self._env.task._candidates)),
      min_summary = np.array([0.0] + [4.5, -1.6, -1.6] * len(self._env.task._candidates)),
      max_trajectory = np.array([1.0, 0.30, 1.2, 1.0, 0.5, 0.5, 5.0, 1.6, 1.6] + [2.0,] * len(self._env.task._candidates)),
      min_trajectory = np.array([0.0, -0.10, 0.6, 0.8, -0.5, -0.5, 4.5, -1.6, 0] + [0.0,] * len(self._env.task._candidates)),
    )

  def simulate(
    self,
    n_param=1,
    min_window_sz=4,
    verbose=False,
    normalize=False,
    fix_start_idx=True,
    fix_progress=0,
  ):
    """
    Simulate target selection behavior
    """
    target_pos_list = list()
    user_params_list = list()
    summary_list = list()
    trajectory_list = list()

    if isinstance(fix_progress, list):
      # if fix_progress is given as a list, we will return the list of tupes for each fix_progress
      for _ in range(len(fix_progress)):
        target_pos_list.append(list())
        user_params_list.append(list())
        summary_list.append(list())
        trajectory_list.append(list())    

    for _ in tqdm(range(n_param), disable=not verbose):
      full_len = 0 
      while full_len < min_window_sz:
        target_pos, user_params, summary, trajectory = self._simulate_trial(normalize=normalize)
        full_len = len(trajectory)

        if isinstance(fix_progress, list):
          window_size_list = [round(full_len * p) for p in fix_progress]

          for i, window_size in enumerate(window_size_list):
            for start_idx in [0] if fix_start_idx else range(full_len-window_size+1):
              target_pos_list[i].append(target_pos)
              user_params_list[i].append(user_params)
              summary_list[i].append(summary)
              trajectory_list[i].append(trajectory[start_idx:start_idx+window_size])

        else:
          if not fix_progress:
            # Data augmentation with different window sizes from the same trajectory
            window_size_list = np.random.choice(
              np.arange(min_window_sz, full_len+1),
              (full_len-min_window_sz+1) // 2,
              replace=False,
            )
          else:
            assert fix_progress <= 1.0
            fixed_window_size = round(full_len * fix_progress)
            window_size_list = [fixed_window_size,]

          for window_size in window_size_list: # range(min_window_sz, full_len+1):
            for start_idx in [0] if fix_start_idx else range(full_len-window_size+1):
              target_pos_list.append(target_pos)
              user_params_list.append(user_params)
              summary_list.append(summary)
              trajectory_list.append(trajectory[start_idx:start_idx+window_size])


    if isinstance(fix_progress, list):
      return [
        [
          np.array(target_pos_list[i]),
          np.array(user_params_list[i]),
          np.array(summary_list[i]),
          np.array(trajectory_list[i], dtype=object)
        ]
      for i in range(len(fix_progress))]
    else:
      return (
        np.array(target_pos_list),
        np.array(user_params_list),
        np.array(summary_list),
        np.array(trajectory_list, dtype=object)
      )

  def _simulate_trial(self, normalize=False):
    """
    Simulate a single trial
    """
    flag = False
    while not flag:
      # Reset environment with given env. parameters
      obs = self._env.reset()
      mujoco.mj_forward(self._env._model, self._env._data)
      state = self._env.get_state()
      user_params = obs["user_params"]
      target_radius = state["target_radius"]
      obj_positions = np.array(state["candidates"] - state["eye_pos"]) * np.array([1, -1, 1]).reshape((1, -1)) # ((# of candidates), 3)
      target_pos = (state["target_pos"] - state["eye_pos"]) * np.array([1, -1, 1])

      ray_trajectory = list()
      require_start = True
      require_perception = True
      done = False
      # Loop until episode ends
      while not done:
        # Get actions from policy
        if require_start or require_perception:
          action, _ = self.pointing_model.predict(obs, deterministic=True)
        else:
          action, _ = self.model.predict(obs, deterministic=True)

        # Take a step
        obs, _, done, info = self._env.step(action)
        require_start = info["require_start"]
        require_perception = not info["target_perceived"]

        if not info["require_start"]:
          # Track trajectory
          state = self._env.get_state()
          ray_trajectory.append(np.concatenate([
            state["ray_origin_pos"] * np.array([1, -1, 1]),
            state["ray_vector"] * np.array([1, -1, 1]),
          ]))
      
      if len(ray_trajectory) > 0:
        flag = True

    ray_trajectory = np.array(ray_trajectory)
    projected_pos = self.projection_of_ray(
      ray_trajectory[:, :3],
      ray_trajectory[:, 3:],
    )
    obj_distances = self.distance_to_objects(
      ray_trajectory[:, :3],
      ray_trajectory[:, 3:],
      np.array(state["candidates"]) * np.array([1, -1, 1]).reshape((1, -1)),
    )
    trajectory = np.concatenate([
      ray_trajectory,
      projected_pos,
      obj_distances,
    ], axis=-1)

    summary = np.concatenate([
      np.array([target_radius,]),
      np.array(obj_positions).flatten(),
    ])
    trajectory = np.array(trajectory)

    if normalize:
      return self.normalize(target_pos, user_params, summary, trajectory)
    else:
      return target_pos, user_params, summary, trajectory
  
  def projection_of_ray(self, pos_arr, dir_arr, center=np.array([0.0, 0.0, 1.2]), target_dist=5.0):
    # Track the position of the projected ray on the same-distance surface
    # Ray P = pos + dir * t
    # Sphere: ||P - O|| = target_dist

    # Quadratic formula coefficients
    A = np.sum(dir_arr**2, axis=1)
    B = 2 * np.sum((pos_arr-center)*dir_arr, axis=1)
    C = np.sum((pos_arr-center)**2, axis=1) - target_dist**2

    # Discriminant
    D = B**2 - 4*A*C
    t = (-B + np.sqrt(D)) / (2*A)
    return pos_arr + t.reshape((-1, 1)) * dir_arr
  
  def distance_to_objects(self, pos_arr, dir_arr, obj_pos_arr):
    # Normalize the direction vectors
    ray_dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=-1, keepdims=True)

    # Reshape the arrays for broadcasting:
    ray_pos_arr_b = np.expand_dims(pos_arr, axis=1)
    ray_dir_arr_b = np.expand_dims(ray_dir_arr, axis=1)
    obj_pos_arr_b = np.expand_dims(obj_pos_arr, axis=0)

    diffs = obj_pos_arr_b - ray_pos_arr_b
    dots = np.sum(diffs * ray_dir_arr_b, axis=-1)
    projs = np.sum(diffs * ray_dir_arr_b, axis=-1, keepdims=True) * ray_dir_arr_b

    orths = diffs - projs
    distances = np.linalg.norm(orths, axis=-1)
    distances[dots < 0] = np.linalg.norm(diffs, axis=-1)[dots < 0]
    return distances
  
  def normalize(self, target_pos, user_params, summary, trajectory):
    return (
      (target_pos - self.norm_info["min_target_pos"]) / \
        (self.norm_info["max_target_pos"] - self.norm_info["min_target_pos"]) * 2 - 1,
      (user_params - self.norm_info["min_user_params"]) / \
        (self.norm_info["max_user_params"] - self.norm_info["min_user_params"]) * 2 - 1,
      (summary - self.norm_info["min_summary"]) / \
        (self.norm_info["max_summary"] - self.norm_info["min_summary"]) * 2 - 1,
      (trajectory - self.norm_info["min_trajectory"]) / \
        (self.norm_info["max_trajectory"] - self.norm_info["min_trajectory"]) * 2 - 1,
    )