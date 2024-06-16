from ...base import BaseModule
from ....utils.functions import parent_path
from ..encoders import one_layer

import numpy as np


class BasicWithEndEffectorPosition(BaseModule):

  def __init__(self, model, data, bm_model, end_effector, **kwargs):
    """ Initialise a new `BasicWithEndEffectorPosition`. Represents proprioception through joint angles, velocities,
    and accelerations, and muscle activation states, and an end effector global position.

    Args:
      model: Mujoco model instance of the simulator.
      data: Mujoco data instance of the simulator.
      bm_model: An instance inheriting from uitb.bm_models.base.BaseBMModel class.
      end_effector (list): A list with first element representing type of mujoco element (geom, body, site), and second
        element is the name of the element
      kwargs: may contain "rng" seed
    """
    super().__init__(model, data, bm_model, **kwargs)
    if not isinstance(end_effector, list) and len(end_effector) != 2:
      raise RuntimeError("end_effector must be a list of size two")
    self._end_effector = end_effector

  @staticmethod
  def insert(task, **kwargs):
    pass

  def get_observation(self, model, data, info=None):

    jnt_idx = self._bm_model.independent_joints

    # Normalise qpos
    jnt_range = model.jnt_range[jnt_idx]
    qpos = data.qpos[jnt_idx].copy()
    qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
    qpos = (qpos - 0.5) * 2

    # Get qvel, qacc
    qvel = data.qvel[jnt_idx].copy()
    qacc = data.qacc[jnt_idx].copy()

    # Get end-effector position; not normalised
    ee_position = getattr(data, self._end_effector[0])(self._end_effector[1]).xpos.copy()

    # Normalise act
    # act = (data.act.copy() - 0.5) * 2 # for muscle control
    act = data.ctrl.copy()

    # jnt_idx (0, 1, 2): r_z, r_x, r_y
    max_degree = 0.0698  # -4 ~ +4 degree 
    torso_pos = data.qpos[[0, 1, 2]].copy() / max_degree

    # Proprioception features
    proprioception = np.concatenate([qpos, qvel, qacc, ee_position, act, torso_pos])
    return proprioception

  def _get_state(self, model, data):
    state = {f"{self._end_effector[1]}_xpos": getattr(data, self._end_effector[0])(self._end_effector[1]).xpos.copy(),
             f"{self._end_effector[1]}_xmat": getattr(data, self._end_effector[0])(self._end_effector[1]).xmat.copy()}
    return state

  @property
  def encoder(self):
    return one_layer(observation_shape=self._observation_shape, out_features=128)