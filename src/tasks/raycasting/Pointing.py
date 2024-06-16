import numpy as np
import mujoco

from .Raycasting import Raycasting
from .reward_functions import HitMissShaping


class Pointing(Raycasting):

  def __init__(self, model, data, origin, ray_origin, ray_target, **kwargs):
    super().__init__(
      model=model,
      data=data,
      origin=origin,
      ray_origin=ray_origin,
      ray_target=ray_target,
      **kwargs
    )
    # Define dwell time for pointing the start object
    self._dwell_time = 0.15
    self._required_dwell_count = self._dwell_time * self._action_sample_freq
    self._count_for_dwell = 0

    self._reward_function = HitMissShaping(k=2.0, amp=0.1, miss_w=self._miss_w, early_stop=self._with_early_stop)

  def _update(self, model, data):
    # Set 'requre_start' True always
    self._info["require_start"] = True 
    
    # Set some defaults
    self._update_ray(data)
    finished = False

    # Agent should pass the ray through the start object to start a trial
    if self._info["require_start"]:
      self._pre_trial_steps += 1

      dist_to_start = self.distance_between_ray_and_point(
        self._info["ray_origin_pos"],
        self._info["ray_vector"],
        model.body(f"obj_start").pos[:]
      )

      if dist_to_start <= self._start_radius * self._scale:
        self._spawn_target(model, data)
        self._count_for_dwell += 1
      else:
        self._count_for_dwell = 0
        if self._pre_trial_steps >= self._max_steps:
          self._info["timeout"] = True

      if self._count_for_dwell >= self._required_dwell_count:
        self._info["start_object_hit"] = True
        self._info["target_hit"] = True

      reward = self._reward_function.get(
        self,
        dist_to_start/self._scale - self._start_radius,
        data
      )
      info = self._info.copy()

    # Trial ends
    if self._info["target_hit"] or self._info["timeout"]:
      self._reset_trial(model, data)

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      finished = True

    mujoco.mj_forward(model, data)

    return reward, finished, info
  
  def _reset_trial(self, model, data):
    self._count_for_dwell = 0
    super()._reset_trial(model, data)
  
  def _reset(self, model, data):
    self._count_for_dwell = 0
    super()._reset(model, data, set_initial_pose=True)