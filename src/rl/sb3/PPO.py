import os
import importlib
import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F
from gym import spaces

from stable_baselines3 import PPO as PPO_sb3
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.distributions import DiagGaussianDistribution

from ..base import BaseRLModel
from .callbacks import EvalCallback, EvalCallbackForPointAndClick
from ...utils.functions import strtime, natural_sort
from ... import Simulator

from .policies import MultiInputACPolicyTanhWithDiscriminator
from .discriminator import Discriminator


class PPO(BaseRLModel):

  def __init__(self, simulator, model_name=None):
    super().__init__()

    rl_config = self.load_config(simulator)
    run_parameters = simulator.run_parameters
    self.simulator_folder = simulator.simulator_folder
    self.model_name = model_name if model_name is not None else strtime()

    # Get total timesteps
    self.total_timesteps = rl_config["total_timesteps"]

    # Initialise parallel envs
    self.parallel_envs = make_vec_env(simulator.__class__, n_envs=rl_config["num_workers"],
                                 seed=run_parameters.get("random_seed", None), vec_env_cls=SubprocVecEnv,
                                 env_kwargs={"simulator_folder": self.simulator_folder})

    # Add feature and stateful information encoders to policy_kwargs
    encoders = simulator.perception.encoders.copy()
    if simulator.task.get_stateful_information_space_params() is not None:
      encoders["stateful_information"] = simulator.task.stateful_information_encoder
    if simulator.bm_model._with_user_params:
      encoders["user_params"] = None
    rl_config["policy_kwargs"]["features_extractor_kwargs"] = {"extractors": encoders}
    self._task_level = simulator.task._level
    self._fix_shoulder = simulator.task._fix_shoulder

    # Initialise model
    if rl_config["policy_type"] is MultiInputACPolicyTanhWithDiscriminator:
      self.use_diayn = True
      self.model = PPO_with_DIAYN_sb3(rl_config["policy_type"], self.parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                          tensorboard_log=self.simulator_folder, n_steps=rl_config["nsteps"], gamma=rl_config["gamma"],
                          batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"], use_sde=rl_config["use_sde"],
                          learning_rate=rl_config["lr"], device=rl_config["device"], ent_coef=rl_config["ent_coef"],
                          beta_reward=rl_config["beta_reward"], beta_loss=rl_config["beta_loss"])

    else:
      self.use_diayn = False
      self.model = PPO_sb3(rl_config["policy_type"], self.parallel_envs, verbose=1, policy_kwargs=rl_config["policy_kwargs"],
                          tensorboard_log=self.simulator_folder, n_steps=rl_config["nsteps"], gamma=rl_config["gamma"],
                          batch_size=rl_config["batch_size"], target_kl=rl_config["target_kl"], use_sde=rl_config["use_sde"],
                          learning_rate=rl_config["lr"], device=rl_config["device"], ent_coef=rl_config["ent_coef"])


    # Create a checkpoint callback
    save_freq = rl_config["save_freq"] // rl_config["num_workers"]
    checkpoint_folder = os.path.join(self.simulator_folder, "../../data/simulator_models", self.model_name)
    self.checkpoint_callback = CheckpointCallback(save_freq=save_freq,
                                                  save_path=checkpoint_folder,
                                                  name_prefix="model")

    # Get callbacks as a list
    self.callbacks = [*simulator.callbacks.values()]

    # Create an evaluation callback
    eval_env = Simulator.get(rl_config["eval_path"])
    if simulator.config["simulation"]["task"]["cls"] == "Click":
      eval_callback_cls = EvalCallbackForPointAndClick
    else:
      eval_callback_cls = EvalCallback

    self.eval_callback = EveryNTimesteps(
      n_steps=rl_config["save_freq"],
      callback=eval_callback_cls(
        eval_env,
        num_eval_episodes=1,
        rl_config=rl_config,
        video_path=os.path.join(self.simulator_folder, "../../results/videos", self.model_name),
      )
    )
    self.rl_config = rl_config

  def load_config(self, simulator):
    config = simulator.config["rl"]

    # Need to translate strings into classes
    config["policy_type"] = simulator.get_class("rl.sb3", config["policy_type"])

    if "activation_fn" in config["policy_kwargs"]:
      mods = config["policy_kwargs"]["activation_fn"].split(".")
      config["policy_kwargs"]["activation_fn"] = getattr(importlib.import_module(".".join(mods[:-1])), mods[-1])

    config["policy_kwargs"]["features_extractor_class"] = \
      simulator.get_class("rl.sb3", config["policy_kwargs"]["features_extractor_class"])

    if "lr" in config:
      if isinstance(config["lr"], dict):
        config["lr"] = simulator.get_class("rl.sb3", config["lr"]["function"])(**config["lr"]["kwargs"])

    return config

  def learn(self, wandb_callback=None):
    self.compute_total_params()

    callbacks = list()
    if wandb_callback:
      callbacks = [wandb_callback]
    else:
      callbacks = list()

    callbacks.extend([
      self.checkpoint_callback,
      self.eval_callback,
      *self.callbacks
    ])
        
    self.model.learn(
      total_timesteps=self.total_timesteps,
      callback=callbacks
    )

  def load(self):
    ckpt_dir = os.path.join(self.simulator_folder, "../../data/simulator_models", self.model_name)
    loaded = False
    if os.path.exists(ckpt_dir):
      if len(os.listdir(ckpt_dir)) > 0:
        loaded = True
        files = natural_sort(os.listdir(ckpt_dir))
        model_file = files[-1]

        custom_objects = {"policy_kwargs": self.rl_config["policy_kwargs"]}
        loaded_model = self.model.load(os.path.join(ckpt_dir, model_file), 
                                       custom_objects=custom_objects)
        self.model.policy.load_state_dict(loaded_model.policy.state_dict())

        if model_file == "base.zip":
          with th.no_grad():
            self.model.policy.log_std.fill_(0.0) # Initialize the std with 1.0
        
        print("Loaded from:", os.path.join(self.model_name, model_file))
        self.compute_total_params()

    if not loaded:
      # Use a pre-trained "pointing" model for soft-loading
      if self._fix_shoulder:
        pointing_ckpt_name = "fixed_pointing"
      elif self._task_level in [1, 2]:
        pointing_ckpt_name = "pointing_dense"
      else:
        pointing_ckpt_name = "pointing_wide"

      soft_ckpt_dir = os.path.join(self.simulator_folder, "../../data/simulator_models", pointing_ckpt_name)
      if os.path.exists(soft_ckpt_dir):
        if len(os.listdir(soft_ckpt_dir)) > 0:
          files = natural_sort(os.listdir(soft_ckpt_dir))
          model_file = files[-1]

          assert isinstance(self.model.policy.action_dist, DiagGaussianDistribution)
          loaded_model = self.model.load(os.path.join(soft_ckpt_dir, model_file))
          self.model.policy.load_state_dict(loaded_model.policy.state_dict())

          with th.no_grad():
            self.model.policy.log_std.fill_(0.0) # Initialize the std with 1.0
          print("(Soft) loaded from:", os.path.join(pointing_ckpt_name, model_file))

  def compute_total_params(self, verbose=True):
    params = 0
    for p in list(self.model.policy.parameters()):
      params += np.prod(list(p.size()))
    self.total_params = params
    if verbose:
      print(f"[ Total trainable parameters: {self.total_params} ]")


class PPO_with_DIAYN_sb3(PPO_sb3):
  def __init__(
    self,
    policy,
    env,
    learning_rate = 3e-4,
    n_steps = 2048,
    batch_size = 64,
    n_epochs = 10,
    gamma = 0.99,
    gae_lambda = 0.95,
    clip_range = 0.2,
    clip_range_vf = None,
    normalize_advantage = True,
    ent_coef = 0.0,
    vf_coef = 0.5,
    max_grad_norm = 0.5,
    use_sde = False,
    sde_sample_freq = -1,
    target_kl = None,
    tensorboard_log = None,
    policy_kwargs = None,
    verbose = 0,
    seed = None,
    device = "auto",
    _init_setup_model = True,
    beta_loss = 1.0,
    beta_reward = 0.1,
  ):
    """
    PPO with DIAYN loss: it encourages the agent to explore different "skills" according to given user params
    """
    super().__init__(
      policy,
      env,
      learning_rate,
      n_steps,
      batch_size,
      n_epochs,
      gamma,
      gae_lambda,
      clip_range,
      clip_range_vf,
      normalize_advantage,
      ent_coef,
      vf_coef,
      max_grad_norm,
      use_sde,
      sde_sample_freq = sde_sample_freq,
      target_kl = target_kl,
      tensorboard_log = tensorboard_log,
      policy_kwargs = policy_kwargs,
      verbose = verbose,
      seed = seed,
      device = device,
      _init_setup_model = _init_setup_model,
    )
    self.beta_reward = beta_reward
    self.beta_loss = beta_loss

  def _setup_model(self) -> None:
    super()._setup_model()

    fake_proprio = self.observation_space.sample()["proprioception"]
    fake_z = self.observation_space.sample()["user_params"]
    self.discriminator = Discriminator(
      in_sz=fake_proprio.shape[0],
      out_sz=fake_z.shape[0],
      device=self.device,
    )

  def collect_rollouts(
    self,
    env,
    callback,
    rollout_buffer,
    n_rollout_steps,
  ):
    """
    Added DIAYN reward compared to the native stable-baselines3 PPO
    """
    assert self._last_obs is not None, "No previous observation was provided"
    # Switch to eval mode (this affects batch norm / dropout)
    self.policy.set_training_mode(False)

    n_steps = 0
    rollout_buffer.reset()
    # Sample new weights for the state dependent exploration
    if self.use_sde:
      self.policy.reset_noise(env.num_envs)

    callback.on_rollout_start()

    while n_steps < n_rollout_steps:
      if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
        # Sample a new noise matrix
        self.policy.reset_noise(env.num_envs)

      with th.no_grad():
        # Convert to pytorch tensor or to TensorDict
        obs_tensor = obs_as_tensor(self._last_obs, self.device)
        actions, values, log_probs = self.policy(obs_tensor)
      actions = actions.cpu().numpy()

      # Rescale and perform action
      clipped_actions = actions
      # Clip the actions to avoid out of bound error
      if isinstance(self.action_space, spaces.Box):
        clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

      new_obs, rewards, dones, infos = env.step(clipped_actions)

      with th.no_grad():
        log_prob_z_given_s = self.discriminator.loss(self._last_obs)
        rewards += self.beta_reward * log_prob_z_given_s.mean(dim=-1).cpu().numpy()

      self.num_timesteps += env.num_envs

      # Give access to local variables
      callback.update_locals(locals())
      if callback.on_step() is False:
        return False

      self._update_info_buffer(infos)
      n_steps += 1

      if isinstance(self.action_space, spaces.Discrete):
        # Reshape in case of discrete action
        actions = actions.reshape(-1, 1)

      # Handle timeout by bootstraping with value function
      # see GitHub issue #633
      for idx, done in enumerate(dones):
        if (
          done
          and infos[idx].get("terminal_observation") is not None
          and infos[idx].get("TimeLimit.truncated", False)
        ):
          terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
          with th.no_grad():
            terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
          rewards[idx] += self.gamma * terminal_value

      rollout_buffer.add(
        self._last_obs,  # type: ignore[arg-type]
        actions,
        rewards,
        self._last_episode_starts,  # type: ignore[arg-type]
        values,
        log_probs,
      )
      self._last_obs = new_obs  # type: ignore[assignment]
      self._last_episode_starts = dones

    with th.no_grad():
      # Compute value for the last timestep
      values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

    rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    callback.on_rollout_end()

    return True

  def train(self):
    """
    Update policy using the currently gathered rollout buffer.
    """
    # Switch to train mode (this affects batch norm / dropout)
    self.policy.set_training_mode(True)
    # Update optimizer learning rate
    self._update_learning_rate(self.policy.optimizer)
    # Compute current clip range
    clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
    # Optional: clip range for the value function
    if self.clip_range_vf is not None:
      clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

    entropy_losses, diayn_losses = [], []
    pg_losses, value_losses = [], []
    clip_fractions = []

    continue_training = True
    # train for n_epochs epochs
    for epoch in range(self.n_epochs):
      approx_kl_divs = []
      # Do a complete pass on the rollout buffer
      for rollout_data in self.rollout_buffer.get(self.batch_size):
        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
          # Convert discrete action from float to long
          actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
          self.policy.reset_noise(self.batch_size)

        values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        values = values.flatten()
        # Normalize advantage
        advantages = rollout_data.advantages
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
          advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        if self.clip_range_vf is None:
          # No clipping
          values_pred = values
        else:
          # Clip the difference between old and new value
          # NOTE: this depends on the reward scaling
          values_pred = rollout_data.old_values + th.clamp(
              values - rollout_data.old_values, -clip_range_vf, clip_range_vf
          )
        # Value loss using the TD(gae_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, values_pred)
        value_losses.append(value_loss.item())

        # Entropy loss favor exploration
        if entropy is None:
          # Approximate entropy when no analytical form
          entropy_loss = -th.mean(-log_prob)
        else:
          entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        discriminator_loss = -th.mean(self.discriminator.loss(rollout_data.observations))
        diayn_losses.append(discriminator_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

        # Calculate approximate form of reverse KL Divergence for early stopping
        # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
        # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
        # and Schulman blog: http://joschu.net/blog/kl-approx.html
        with th.no_grad():
          log_ratio = log_prob - rollout_data.old_log_prob
          approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
          approx_kl_divs.append(approx_kl_div)

        if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
          continue_training = False
          if self.verbose >= 1:
            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
          break

        # Optimization step
        self.discriminator.optimizer.zero_grad()
        discriminator_loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.max_grad_norm)
        self.discriminator.optimizer.step()

        # Optimization step
        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Clip grad norm
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

      self._n_updates += 1
      if not continue_training:
        break

    explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

    # Logs
    self.logger.record("train/entropy_loss", np.mean(entropy_losses))
    self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
    self.logger.record("train/value_loss", np.mean(value_losses))
    self.logger.record("train/diayn_loss", np.mean(diayn_losses))
    self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
    self.logger.record("train/clip_fraction", np.mean(clip_fractions))
    self.logger.record("train/loss", loss.item())
    self.logger.record("train/explained_variance", explained_var)
    if hasattr(self.policy, "log_std"):
      self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
    self.logger.record("train/clip_range", clip_range)
    if self.clip_range_vf is not None:
      self.logger.record("train/clip_range_vf", clip_range_vf)
