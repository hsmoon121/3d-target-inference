from .Raycasting import Raycasting
from .reward_functions import HitMissLinear

class RaycastingEval(Raycasting):

  def __init__(self, model, data, origin, ray_origin, ray_target, **kwargs):
    super().__init__(
      model=model,
      data=data,
      origin=origin,
      ray_origin=ray_origin,
      ray_target=ray_target,
      **kwargs
    )
    self._max_steps = self._action_sample_freq * 10
    self._max_trials = len(self._candidates)
    self._reward_function = HitMissLinear(k=2.0, amp=0.1, miss_w=self._miss_w)
    self._reward_function.update(var=1.0)

  def _reset_trial(self, model, data):
    super()._reset_trial(model, data)

    # print summary at last trial
    if self._trial_idx >= self._max_trials:
      print(f"=== Error rate: {self._info['error_rate']:.2f} (%), Completion time: {self._info['completion_time']:.2f} (s)")

  def _reset(self, model, data):
    super()._reset(model, data, set_initial_pose=True)
    
  def _choose_target(self):
    # sequential targets
    return self._candidates[self._trial_idx]