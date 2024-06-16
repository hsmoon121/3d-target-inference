import numpy as np
import torch
import mujoco
import os
import pandas as pd
import warnings
import wandb
import skvideo.io

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.utils.functions import grab_pip_image, natural_sort


class LinearStdDecayCallback(BaseCallback):
    """
    Linearly decaying standard deviation

    :param initial_log_value: Log initial standard deviation value
    :param threshold: Threshold for progress remaining until decay begins
    :param min_value: Minimum value for standard deviation
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, initial_log_value, threshold, min_value, verbose=0):
      super(LinearStdDecayCallback, self).__init__(verbose)
      self.initial_value = np.exp(initial_log_value)
      self.threshold = threshold
      self.min_value = min_value

    def _on_rollout_start(self) -> None:
      progress_remaining = self.model._current_progress_remaining
      if progress_remaining > self.threshold:
        pass
      else:
        new_std = self.min_value + (progress_remaining/self.threshold) * (self.initial_value-self.min_value)
        self.model.policy.log_std.data = torch.tensor(np.log(new_std)).float()

    def _on_training_start(self) -> None:
      pass

    def _on_step(self) -> bool:
      return True

    def _on_rollout_end(self) -> None:
      pass

    def _on_training_end(self) -> None:
      pass


class LinearCurriculum(BaseCallback):
  """
  A callback to implement linear curriculum for one parameter

  :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
  """
  def __init__(self, name, start_value, end_value, end_timestep, start_timestep=0, verbose=0):
    super().__init__(verbose)
    self.name = name
    self.variable = start_value
    self.start_value = start_value
    self.end_value = end_value
    self.start_timestep = start_timestep
    self.end_timestep = end_timestep
    self.coeff = (end_value - start_value) / (end_timestep - start_timestep)

  def value(self):
    return self.variable

  def update(self, num_timesteps):
    if num_timesteps <= self.start_timestep:
      self.variable = self.start_value
    elif self.end_timestep >= num_timesteps > self.start_timestep:
      self.variable = self.start_value + self.coeff * (num_timesteps - self.start_timestep)
    else:
      self.variable = self.end_value

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:
    pass

  def _on_step(self) -> bool:
    return True

  def _on_rollout_end(self) -> None:
    self.update(self.num_timesteps)
    self.training_env.env_method("update_reward", self.variable) # From 0 to 1
    self.logger.record("Charts/reward_variable", self.variable)

  def _on_training_end(self) -> None:
    pass
  

class EvalCallback(BaseCallback):
  """
  A custom callback that derives from ``BaseCallback``.

  :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
  """

  def __init__(self, env, num_eval_episodes, rl_config, video_path, verbose=0):
    super().__init__(verbose)
    self.env = env
    self.num_eval_episodes = num_eval_episodes
    self.video_path = video_path
    os.makedirs(self.video_path, exist_ok=True)

  def _on_training_start(self) -> None:
    pass

  def _on_rollout_start(self) -> None:
    pass

  def _on_step(self) -> bool:
    # Run a few episodes to evaluate progress, with and without deterministic actions
    det_info = self.evaluate(deterministic=True)
    sto_info = self.evaluate(deterministic=False)

    # Log evaluations
    self.logger.record("evaluate/deterministic/ep_rew_mean", det_info[0])
    self.logger.record("evaluate/deterministic/ep_len_mean", det_info[1])

    self.logger.record("evaluate/stochastic/ep_rew_mean", sto_info[0])
    self.logger.record("evaluate/stochastic/ep_len_mean", sto_info[1])

    if hasattr(self.env.task, "get_summary_stat"):
      for key in det_info[2]:
        self.logger.record(f"evaluate/deterministic/{key}", det_info[2][key])
      for key in sto_info[2]:
        self.logger.record(f"evaluate/stochastic/{key}", sto_info[2][key])

    params = 0
    for p in list(self.model.policy.parameters()):
      params += np.prod(list(p.size()))
    total_params = params
    self.logger.record("Charts/total_params", total_params)
    self.logger.dump(step=self.num_timesteps)
    return True

  def _on_rollout_end(self) -> None:
    pass

  def _on_training_end(self) -> None:
    pass

  def evaluate(self, deterministic):

    rewards = np.zeros((self.num_eval_episodes,))
    episode_lengths = np.zeros((self.num_eval_episodes,))
    imgs = list()

    if hasattr(self.env.task, "get_summary_stat"):
      ct_arr = np.zeros((self.num_eval_episodes,))
      err_arr = np.zeros((self.num_eval_episodes,))

    for i in range(self.num_eval_episodes):

      obs = self.env.reset()
      mujoco.mj_forward(self.env._model, self.env._data)

      done = False
      while not done:
        action, _ = self.model.predict(obs, deterministic=deterministic)
        obs, r, done, info = self.env.step(action)
        rewards[i] += r
        episode_lengths[i] += 1

        if i == 0 and episode_lengths[i] <= 1000: # record only 1000 steps because of memory issue
          imgs.append(grab_pip_image(self.env))

      if hasattr(self.env.task, "get_summary_stat"):
        ct_arr[i], err_arr[i] = self.env.task.get_summary_stat()

    # Write the video
    subfix = "det" if deterministic else "sto"
    video_file = f"evaluate_{self.num_timesteps}step_{subfix}.mp4"

    with warnings.catch_warnings():
      warnings.filterwarnings("ignore",category=DeprecationWarning)
      skvideo.io.vwrite(
          os.path.join(self.video_path, video_file),
          np.asarray(imgs),
          outputdict={"-pix_fmt": "yuv420p"}
      )

    if self.n_calls % 10 == 0 or self.n_calls <= 1:
      if wandb.run is not None and deterministic:
        wandb_frames = np.transpose(np.asarray(imgs), (0, 3, 1, 2))[:, :, ::2, ::2]
        wandb.log({"video": wandb.Video(wandb_frames, fps=20)})

    if hasattr(self.env.task, "get_summary_stat"):
      additional_log = dict(
        avg_completion_time = np.mean(ct_arr),
        avg_error_rate = np.mean(err_arr)
      )
      return np.mean(rewards), np.mean(episode_lengths), additional_log
    else:
      return np.mean(rewards), np.mean(episode_lengths)
    

class EvalCallbackForPointAndClick(EvalCallback):
  def __init__(self, env, num_eval_episodes, rl_config, video_path, verbose=0):
    super().__init__(env, num_eval_episodes, rl_config, video_path, verbose)

    # Use a pre-trained "pointing" model
    if self.env.task._fix_shoulder:
      pointing_ckpt_name = "fixed_pointing"
    elif self.env.task._level in [1, 2]:
      pointing_ckpt_name = "pointing_dense"
    else:
      pointing_ckpt_name = "pointing_wide"

    pointing_ckpt_dir = os.path.join(self.video_path, "../../../data/simulator_models", pointing_ckpt_name)
    assert os.path.exists(pointing_ckpt_dir), "Pointing model should be trained before Click model"

    files = natural_sort(os.listdir(pointing_ckpt_dir))
    model_file = files[-1]

    custom_objects = {"policy_kwargs": rl_config["policy_kwargs"]}
    self.pointing_model = PPO.load(os.path.join(pointing_ckpt_dir, model_file), custom_objects=custom_objects)
    print(f'Loading [ pointing ] model: {os.path.join(pointing_ckpt_name, model_file)}\n')

  def evaluate(self, deterministic):
    rewards = np.zeros((self.num_eval_episodes,))
    episode_lengths = np.zeros((self.num_eval_episodes,))
    imgs = list()
    trial_summ = list()

    if deterministic:
      self.performance_df = pd.DataFrame()

    if hasattr(self.env.task, "get_summary_stat"):
      ct_arr = np.zeros((self.num_eval_episodes,))
      err_arr = np.zeros((self.num_eval_episodes,))

    for i in range(self.num_eval_episodes):

      obs = self.env.reset()
      mujoco.mj_forward(self.env._model, self.env._data)

      for target_idx in range(len(self.env.task._candidates)):

        require_start = True
        require_perception = True
        done = False

        while not done:
          if require_start or require_perception:
            action, _ = self.pointing_model.predict(obs, deterministic=deterministic)
          else:
            action, _ = self.model.predict(obs, deterministic=deterministic)

          obs, r, done, info = self.env.step(action)
          state = self.env.get_state()

          rewards[i] += r
          episode_lengths[i] += 1
          
          require_start = info["require_start"]
          require_perception = not info["target_perceived"]
          done = state["trial_idx"] >= target_idx + 1

          if done:
            success = info["target_hit"]
          else:
            target_pos = state["target_pos"]
            target_width = state["target_radius"] * 2
            time_elapsed = (self.env.task._trial_steps + 1) / self.env.task._action_sample_freq

          if i == 0 and episode_lengths[i] <= 1000: # record only 1000 steps because of memory issue
            imgs.append(grab_pip_image(self.env))

        trial_summ.append([
          -target_pos[1],
          target_pos[2],
          target_pos[0],
          target_width,
          time_elapsed,
          success
        ])

      if hasattr(self.env.task, "get_summary_stat"):
        ct_arr[i], err_arr[i] = self.env.task.get_summary_stat()

    # Write the video
    subfix = "det" if deterministic else "sto"
    video_file = f"evaluate_{self.num_timesteps}step_{subfix}.mp4"

    with warnings.catch_warnings():
      warnings.filterwarnings("ignore",category=DeprecationWarning)
      skvideo.io.vwrite(
          os.path.join(self.video_path, video_file),
          np.asarray(imgs),
          outputdict={"-pix_fmt": "yuv420p"}
      )

    if self.n_calls % 10 == 0 or self.n_calls <= 1:
      if wandb.run is not None and deterministic:
        wandb_frames = np.transpose(np.asarray(imgs), (0, 3, 1, 2))[:, :, ::2, ::2]
        wandb.log({"video": wandb.Video(wandb_frames, fps=20)})

    if hasattr(self.env.task, "get_summary_stat"):
      additional_log = dict(
        avg_completion_time = np.mean(ct_arr),
        avg_error_rate = np.mean(err_arr)
      )
      return np.mean(rewards), np.mean(episode_lengths), additional_log
    else:
      return np.mean(rewards), np.mean(episode_lengths)