from .Raycasting import Raycasting


class Click(Raycasting):

  def __init__(self, model, data, origin, ray_origin, ray_target, **kwargs):
    super().__init__(
      model=model,
      data=data,
      origin=origin,
      ray_origin=ray_origin,
      ray_target=ray_target,
      **kwargs
    )
    self._info["require_start"] = False # Set 'requre_start' False always
  
  def _reset_trial(self, model, data):
    super()._reset_trial(model, data)
    self._info["require_start"] = False

  def _reset(self, model, data):
    super()._reset(model, data, set_initial_pose=True)
    self._spawn_target(model, data) # Spawn the target right after reset
    self._info["require_start"] = False