import numpy as np
from abc import ABC, abstractmethod


class BaseFunction(ABC):
  @abstractmethod
  def get(self, env, dist, info):
    pass
  @abstractmethod
  def __repr__(self):
    pass
  

class HitMissLinear(BaseFunction):
  def __init__(self, k=2.0, amp=0.1, miss_w=1.0, early_stop=False):
    self.hit_bonus = 10.0
    self.miss_penalty = miss_w * (-1.0)
    self.timeout_penalty_w = 20.0
    self.time_penalty = -0.05
    self.jerk_penalty = -0.0025
    self.early_stop = early_stop

    self.k = k
    self.amp = amp
    self._var = 0.0

  def update(self, var):
    # By linear curriculum, the self._var will increase from 0.0 to 1.0
    self._var = var

  def get(self, env, dist, data):
    if dist <= 0:
      dist_penalty = 0.0
    else:
      dist_penalty = (np.exp(-dist*self.k) - 1)

    linear_penalty = (1 - self._var) * dist_penalty + self._var * self.time_penalty
    linear_penalty *=  self.amp

    # Calculate penalty based on distance to target surface
    if env._info["require_start"]:
      if env._info["start_object_hit"]:
        task_reward = self.hit_bonus
      elif env._info["timeout"]:
        task_reward = self.timeout_penalty_w * dist_penalty # * linear_penalty
      else:
        task_reward = linear_penalty

    else:
      if env._info["target_hit"]:
        task_reward = self.hit_bonus
      elif env._info["timeout"]:
        task_reward = self.timeout_penalty_w * dist_penalty # * linear_penalty
      elif env._info["wrong_target"] or env._info["target_miss"]:
        if self.early_stop:
          task_reward = (1 - self._var) * (env._info["remaining_step"]) * dist_penalty * self.amp + \
            self._var * self.miss_penalty
        else:
          task_reward = linear_penalty + self.miss_penalty * self._var
      else:
        task_reward = linear_penalty
      
    jerk_reward = (np.linalg.norm(env._info["ray_origin_jerk"])) * self.jerk_penalty * self._var
    return task_reward + jerk_reward
    
  def __repr__(self):
    return "HitMissLinear"
  

class HitMissShaping(HitMissLinear):
  # Reward function where var is always 0.0 (=fully shaped reward)
  def update(self, var):
    self._var = 0.0

  def __repr__(self):
    return "HitMissShaping"