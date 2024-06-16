import math
import numpy as np
import mujoco
import xml.etree.ElementTree as ET

from .reward_functions import HitMissLinear
from ..base import BaseTask


COLOR_HIDDEN = np.array([1.0, 1.0, 1.0, 0.0])
COLOR_UNSELECTED = np.array([1.0, 1.0, 1.0, 1.0])
COLOR_SELECTED = np.array([0.6, 1.0, 0.6, 1.0])
COLOR_TARGET = np.array([0.0, 0.0, 1.0, 1.0])
COLOR_START = np.array([0.3, 0.3, 0.3, 1.0])


class Raycasting(BaseTask):

  def __init__(self, model, data, origin, ray_origin, ray_target, **kwargs):
    super().__init__(model, data, **kwargs)

    # This task requires an origin/ray_origin/ray_target to be defined
    if not isinstance(origin, list) and len(origin) != 2:
      raise RuntimeError("'origin' must be a list with two elements: first defining what type of mujoco element "
                          "it is, and second defining the name")
    self._origin = origin
    self._origin_pos = np.array([0.0, 0.0, 1.0]) # getattr(data, self._origin[0])(self._origin[1]).xpos
    
    if not isinstance(ray_origin, list) and len(ray_origin) != 2:
      raise RuntimeError("'ray_origin' must be a list with two elements: first defining what type of mujoco element "
                          "it is, and second defining the name")
    self._ray_origin = ray_origin

    if not isinstance(ray_target, list) and len(ray_target) != 2:
      raise RuntimeError("'ray_target' must be a list with two elements: first defining what type of mujoco element "
                          "it is, and second defining the name")
    self._ray_target = ray_target

    # Define necessary variables
    self._fix_shoulder = kwargs.get("fix_shoulder", True)
    if self._fix_shoulder:
      self._joint_actuators = [16, 17, 18, 19, 20]
    else:
      self._joint_actuators = [13, 14, 16, 17, 18, 19, 20]
      
    self.level = kwargs.get("level", 1)
    if isinstance(self.level, list):
      self._level = self.level[0]
    else:
      self._level = self.level 

    self._max_trials = kwargs.get("max_trials", 10)
    self._trial_idx = 0
    self._targets_hit = 0
    self._targets_miss = 0
    self._total_steps = 0
    self._initial_qpos = None

    # If "limb_scale" is adjusted, there should be relevant scaling of environmental elements
    # 1) targets & start objects' size & distance should be adjusted in perspective of eye position
    # 2) reward calucation should be done with adjusted distance
    # 3) observatory camera system should be re-adjusted
    self.user_params = dict()
    self._scale = 1.0
    self._miss_w = 1.0
    self._var = 0.5

    # Use early termination if target is not hit in time
    self._trial_steps = 0
    self._pre_trial_steps = 0
    self._max_seconds = 6
    self._max_steps = self._action_sample_freq * self._max_seconds

    # Used for logging states
    self._info = {
      "target_hit": False,
      "wrong_target": False,
      "target_miss": False,
      "pointing_target": False,
      "require_start": True,
      "start_object_hit": False,
      "target_perceived": False,
      "timeout": False,
      "hit_at_last_trial": False,
      "remaining_step": self._max_steps,
      "ray_origin_pos": np.array([0.0, 0.0, 0.0]),
      "ray_origin_vel": np.array([0.0, 0.0, 0.0]),
      "ray_origin_acc": np.array([0.0, 0.0, 0.0]),
      "ray_origin_jerk": np.array([0.0, 0.0, 0.0]),
    }

    # Define a reaction time for click
    self._click_reaction_time_mean = 0.2
    self._click_reaction_time_std = 0.05
    self._click_delay = self._click_reaction_time_mean * self._action_sample_freq
    self._click_decision = False
    self._count_for_click = 0

    # Define a reaction time for target perception
    self._target_reaction_time_mean = 0.2
    self._target_reaction_time_std = 0.05
    self._target_percp_delay = self._target_reaction_time_mean * self._action_sample_freq
    self._count_for_target_percp = 0

    # Do a forward step so stuff like geom and body positions are calculated
    mujoco.mj_forward(model, data)

    # Set camera angle
    self._cam_pos = np.array([-1.6, -1.2, 2.2])
    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "for_testing")] = self._cam_pos
    model.cam_quat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "for_testing")] = np.array(
    [0.718027, 0.4371043, -0.31987, -0.4371043])

    # Define a default reward function
    self._linear_curriculum = kwargs.get("linear_curriculum", False)
    self._with_early_stop = kwargs.get("with_early_stop", True)

    self._reward_function = HitMissLinear(k=2.0, amp=0.1, miss_w=self._miss_w, early_stop=self._with_early_stop)

    # Set grid (task)
    self._with_distractors = kwargs.get("with_distractors", True)
    self._with_perturbation = kwargs.get("with_perturbation", True)
    self.set_grid(model, data)

  def set_grid(self, model, data):
    """
    Levels (different conditions):
    (1) small grid & large target
    (2) small grid & small target
    (3) large grid & large target
    (4) large grid & small target
    """
    assert self._level in [1, 2, 3, 4]

    # Initialize all objects' positions
    for obj_i in range(1, 64):
      model.body(f"obj_{obj_i}").pos[:] = np.array([0.0, 0.0, 0.0])
      model.geom(f"obj_{obj_i}").rgba = COLOR_HIDDEN

    target_dist = 5.0 # meters
    target_width = [0.10, 0.06, 0.10, 0.06][self._level-1] # meters
    spacing = [1.44, 1.44, 6.0, 6.0][self._level-1] # degrees
    self._target_radius = target_width / 2
    self._start_radius = 0.05

    n_row = [7, 7, 7, 7][self._level-1]
    n_column = [7, 7, 9, 9][self._level-1]
    self._n_obj = n_row * n_column

    self._start_pos_from_eyes = [
      [np.sqrt(5.0**2 - 1.2**2), 0.0, -1.2], # [depth, x, y]
      [np.sqrt(5.0**2 - 1.2**2), 0.0, -1.2],
      [5.0, 0.0, 0.0],
      [5.0, 0.0, 0.0],
    ][self._level - 1]
    self._candidates = []

    self.thorax_to_eye = 0.2
    self._eye_pos = self._origin_pos + np.array([0, 0, self.thorax_to_eye])

    model.cam_pos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "for_testing")] = \
      self._eye_pos + self._scale * (self._cam_pos - self._eye_pos)

    self._target_i = 1
    self._prev_highlight_i = 1

    for obj_i in range(1, self._n_obj+1):
      row = (obj_i-1) // n_column
      column = (obj_i-1) % n_column
      phi = -(column - (n_column-1)/2) * spacing * math.pi / 180.0 # azimuth angle
      theta = -(row - (n_row-1)/2) * spacing * math.pi / 180.0 # elevation angle

      depth = target_dist * math.cos(theta) * math.cos(phi)
      x = target_dist * math.cos(theta) * math.sin(phi)
      y = target_dist * math.sin(theta)
      obj_pos = self._eye_pos + np.array([depth, x, y]) * self._scale

      # candidates should not be on the outer-most layer
      if row * (row-n_row+1) * column * (column-n_column+1) != 0:
        # candidates also should not be on the center & inner-most layer in level 3 & 4
        if self._level in [1, 2] or abs(row - (n_row-1)/2) > 1 or abs(column - (n_column-1)/2) > 1:
          self._candidates.append(obj_i)

      model.geom(f"obj_{obj_i}").size = self._target_radius * self._scale
      model.body(f"obj_{obj_i}").pos[:] = obj_pos

      if self._level in [3, 4] and row == (n_row-1)/2 and column == (n_column-1)/2:
        # in level 3 & 4, there is no center object (because of the start object)
        model.body(f"obj_{obj_i}").pos[:] = self._eye_pos + np.array([-5.0, x, y]) * self._scale
        model.geom(f"obj_{obj_i}").rgba = COLOR_HIDDEN
      elif self._with_distractors:
        model.geom(f"obj_{obj_i}").rgba = COLOR_UNSELECTED
      else:
        model.geom(f"obj_{obj_i}").rgba = COLOR_HIDDEN

    # Place a start object
    model.geom(f"obj_start").rgba = COLOR_START
    model.body(f"obj_start").pos[:] = self._eye_pos + np.array(self._start_pos_from_eyes) * self._scale
    model.geom(f"obj_start").size = self._start_radius * self._scale

  def _update(self, model, data):

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
        self._info["start_object_hit"] = True
      elif self._pre_trial_steps >= self._max_steps:
        self._info["timeout"] = True

      reward = self._reward_function.get(
        self,
        dist_to_start/self._scale - self._start_radius,
        data
      )
      info = self._info.copy()

      if self._info["start_object_hit"]:
        self._info["require_start"] = False
        # Spawn a new target location
        self._spawn_target(model, data)

        # Set the target perception delay
        self._info["target_perceived"] = False
        self._target_percp_delay = np.random.normal(
          loc=self._target_reaction_time_mean,
          scale=self._target_reaction_time_std,
        ) * self._action_sample_freq

    else:
      # Update the trial step
      self._trial_steps += 1
      self._total_steps += 1
      self._info["remaining_step"] = self._max_steps - self._trial_steps
      self._info["start_object_hit"] = False
      
      if self._trial_steps >= self._target_percp_delay:
        self._info["target_perceived"] = True
      if self._trial_steps < self._target_percp_delay + 1:
        self._click_decision = False

      # Check if the agent is pointing the right target
      first_hit_obj_i, dist_to_target = self.calculate_ray_hit(data) 
      self._info["pointing_target"] = (first_hit_obj_i == self._target_i)

      # If the agent clicks, start counting the delay period
      if self._click_decision:
        self._count_for_click += 1

      # Clicked after sampled reaction delay
      self._info["target_miss"] = False
      self._info["wrong_target"] = False
      if self._count_for_click >= self._click_delay:
        if self._info["pointing_target"]:
          self._info["target_hit"] = True
          self._targets_hit += 1
        elif first_hit_obj_i is None:
          self._info["target_miss"] = True
          self._targets_miss += 1
        else:
          self._info["wrong_target"] = True
          self._targets_miss += 1

        # if the agent missed any target or objects, it continues the trial
        self._click_decision = False
        self._count_for_click = 0

      if self._trial_steps >= self._max_steps:
        self._info["timeout"] = True

      # Calculate reward; note, inputting distance to surface into reward function,
      # hence distance can be negative if ray is inside target
      reward = self._reward_function.get(
        self,
        dist_to_target/self._scale - self._target_radius,
        data
      )
      info = self._info.copy()

    # Trial ends
    if self._info["target_hit"] or self._info["timeout"]:
      self._reset_trial(model, data)
    elif self._with_early_stop and (self._info["wrong_target"] or self._info["target_miss"]):
      self._reset_trial(model, data)

    # Check if max number trials reached
    if self._trial_idx >= self._max_trials:
      finished = True

    mujoco.mj_forward(model, data)

    return reward, finished, info
  
  def _update_ray(self, data):

    def _data(pointer):
      return getattr(data, pointer[0])(pointer[1])
    
    # Get ray origin and angle configuration
    ray_origin = np.array(_data(self._ray_origin).xpos)
    ray_target = np.array(_data(self._ray_target).xpos)
    ray_vector = (ray_target - ray_origin) / np.linalg.norm(ray_target - ray_origin)

    # while ray_origin is scaled by self._scale
    # ray_vel, ray_acc, ray_jerk are in original scale
    ray_vel = (ray_origin - self._info["ray_origin_pos"]) / self._scale
    ray_acc = ray_vel - self._info["ray_origin_vel"]
    self._info["ray_origin_jerk"] = (ray_acc - self._info["ray_origin_acc"]) * (self._action_sample_freq ** 3)

    self._info["ray_origin_pos"] = ray_origin
    self._info["ray_vector"] = ray_vector
    self._info["ray_origin_vel"] = ray_vel
    self._info["ray_origin_acc"] = ray_acc
  
  def distance_between_ray_and_point(self, ray_origin, ray_direction, point): 
    # Calculate the vector from ray_origin to point
    point_vec = point - ray_origin

    # Project the vector onto ray_direction
    projection_length = np.dot(point_vec, ray_direction)

    # If the projection is behind the ray_origin, the distance is the distance between the ray_origin and the point
    if projection_length < 0:
      return np.linalg.norm(point_vec)

    # Otherwise, find the projected point and calculate the distance between the point and the projected point
    projected_point = ray_origin + projection_length * ray_direction
    distance = np.linalg.norm(projected_point - point)

    return distance

  def calculate_ray_hit(self, data):
    # Find the pointed target
    first_hit_obj_i = None
    target_pos = data.body(f"obj_{self._target_i}").xpos
    dist_to_target = self.distance_between_ray_and_point(
      self._info["ray_origin_pos"],
      self._info["ray_vector"],
      target_pos
    )

    if self._with_distractors:
      min_dist = float("inf")
      for obj_i in range(1, self._n_obj+1):
        obj_pos = data.body(f"obj_{obj_i}").xpos
        dist = self.distance_between_ray_and_point(
          self._info["ray_origin_pos"],
          self._info["ray_vector"],
          obj_pos
        )

        # Check if the ray directs the target
        if dist <= self._target_radius * self._scale:
          dist_org_obj = np.linalg.norm(obj_pos - self._info["ray_origin_pos"])

          # Find the closest object (i.e., first hit) from the ray origin
          if dist_org_obj < min_dist:
            first_hit_obj_i = obj_i
            min_dist = dist_org_obj
    else:
      if dist_to_target <= self._target_radius * self._scale:
        first_hit_obj_i = self._target_i

    return first_hit_obj_i, dist_to_target
  
  def _reset_trial(self, model, data):
    self._trial_idx += 1
    self._trial_steps = 0
    self._pre_trial_steps = 0
    self._count_for_click = 0
    self._click_decision = False
    hit_at_last_trial = self._info["target_hit"]

    if self._trial_idx >= self._max_trials:
      if self._targets_miss + self._targets_hit == 0:
        error_rate = 100
      else:
        error_rate = self._targets_miss / (self._targets_miss + self._targets_hit) * 100
      completion_time = self._total_steps / self._action_sample_freq / self._max_trials

      self._info["error_rate"] = error_rate
      self._info["completion_time"] = completion_time

    else:
      if self._with_perturbation:
        self._perturbation(model, data)

      if isinstance(self.level, list):
        self._level = np.random.choice(self.level)
      self.set_grid(model, data)

      self._set_target(model, data)
      self._info.update({
        "target_hit": False,
        "wrong_target": False,
        "target_miss": False,
        "pointing_target": False,
        "require_start": True,
        "start_object_hit": False,
        "target_perceived": False,
        "timeout": False,
        "hit_at_last_trial": hit_at_last_trial,
        "remaining_step": self._max_steps
      })

  def _reset(self, model, data, set_initial_pose=True):
    # Reset counters
    self._total_steps = 0
    self._trial_steps = 0
    self._pre_trial_steps = 0
    self._trial_idx = 0
    self._targets_hit = 0
    self._targets_miss = 0
    self._count_for_click = 0
    self._click_decision = False

    # Set initial pose from previous episode
    if set_initial_pose:
      self._set_initial_pose(model, data)

    if self._with_perturbation:
      self._perturbation(model, data)

    if isinstance(self.level, list):
      self._level = np.random.choice(self.level)
    self.set_grid(model, data)

    self._set_target(model, data)
    self._info.update({
      "target_hit": False,
      "wrong_target": False,
      "target_miss": False,
      "pointing_target": False,
      "require_start": True,
      "start_object_hit": False,
      "target_perceived": False,
      "timeout": False,
      "hit_at_last_trial": False,
      "remaining_step": self._max_steps,
      "ray_origin_pos": np.array([0.0, 0.0, 0.0]),
      "ray_origin_vel": np.array([0.0, 0.0, 0.0]),
      "ray_origin_acc": np.array([0.0, 0.0, 0.0]),
      "ray_origin_jerk": np.array([0.0, 0.0, 0.0]),
    })

  def _choose_target(self):
    # Sample a target from candidates
    return self._rng.choice(self._candidates)
  
  def _set_target(self, model, data):
    # Get previous target back to default colors
    if self._with_distractors:
      model.geom(f"obj_{self._target_i}").rgba = COLOR_UNSELECTED
    else:
      model.geom(f"obj_{self._target_i}").rgba = COLOR_HIDDEN
    # Set target
    self._target_i = self._choose_target()

  def _spawn_target(self, model, data):
    # Set target colored
    model.geom(f"obj_{self._target_i}").rgba = COLOR_TARGET
    mujoco.mj_forward(model, data)

  def _set_initial_pose(self, model, data):
    qpos = self._rng.uniform(
      low=model.jnt_range[self._joint_actuators][:, 0],
      high=model.jnt_range[self._joint_actuators][:, 1]
    )
    data.qpos[self._joint_actuators] = qpos
    mujoco.mj_forward(model, data)

  def _perturbation(self, model, data):
    # Eye position
    base_pos = np.array([0.0, 0.2, 0.0])
    sampled_pos = (self._rng.random((3,))*2 - 1) * 0.01
    model.body(f"fixed-eye").pos = base_pos + sampled_pos * self._var

    # Torso tilting
    max_degree = 0.0349 / 2 # 0.0349: -2 ~ +2 degree 
    sampled_degree = (self._rng.random((3,))*2 - 1) * max_degree * self._var
    model.equality("r_z_constraint").data[0] = sampled_degree[0]
    model.equality("r_x_constraint").data[0] = sampled_degree[1]
    model.equality("r_y_constraint").data[0] = sampled_degree[2]

    mujoco.mj_forward(model, data)

  def get_stateful_information(self, model, data):
    """
    Observation from task itself:
    1) Whether the agent is pointing the right target
    2) How long the agent has been planning to click (during the delay period)
    3) Whether the agent should pass through the start target
    """
    pointing_status = int(self._info["pointing_target"])
    clicking_status = 0.0 # dummy
    start_status = int(self._info["require_start"])
    return np.array([pointing_status, clicking_status, start_status])

  def _get_state(self, model, data):
    state = dict()
    state["eye_pos"] = self._origin_pos + np.array([0, 0, self.thorax_to_eye])
    state["target_pos"] = \
      (data.body(f"obj_{self._target_i}").xpos.copy() - state["eye_pos"]) / self._scale + state["eye_pos"]
    state["target_radius"] = self._target_radius
    state["trial_idx"] = self._trial_idx
    state["targets_hit"] = self._targets_hit
    state["targets_miss"] = self._targets_miss
    state["candidates"] = [
      (data.body(f"obj_{i}").xpos.copy() - state["eye_pos"]) / self._scale + state["eye_pos"] \
        for i in self._candidates
    ]
    state.update(self._info)
    if "ray_origin_pos" in state:
      state["ray_origin_pos"] = \
        (self._info["ray_origin_pos"] - state["eye_pos"]) / self._scale + state["eye_pos"]
    return state

  def set_ctrl(self, model, data, action):
    random_sample = np.random.random(1)*2 - 1
    if action[0] >= random_sample[0] and not self._click_decision:
      self._click_decision = True
      self._click_delay = np.random.normal(
        loc=self._click_reaction_time_mean,
        scale=self._click_reaction_time_std,
      ) * self._action_sample_freq

  def update_user_param(self, user_params):
    self.user_params = user_params
    if "limb_scale" in user_params:
      self._scale = 1 / user_params["limb_scale"]
    if "miss_w" in user_params:
      self._miss_w = user_params["miss_w"]
    self.update_reward()

  def update_reward(self, var=None):
    if var is not None:
      self._reward_function.update(var)
      self._var = var
    self._reward_function.miss_w = self._miss_w

  def get_summary_stat(self):
    return self._info["completion_time"], self._info["error_rate"]

  @classmethod
  def initialise(cls, task_kwargs):
    n_obj = 63 # Potential maximum number of objects (7x9)

    # Parse xml file
    simulation = ET.parse(cls.get_xml_file())
    simulation_root = simulation.getroot()

    # Create objects
    def create_obj(obj_name):
      obj = ET.Element("body", name=f"obj_{obj_name}", pos="0.0 0.0 0.0")
      obj.append(ET.Element("geom", name=f"obj_{obj_name}", type="sphere", size="0.1", rgba="1.0 1.0 1.0 0.0"))
      return obj
    
    for i in range(n_obj):
      simulation_root.find("worldbody").append(create_obj(f"{i+1}"))
    simulation_root.find("worldbody").append(create_obj("start"))
    return simulation

  @property
  def nu(self):
    """ Return number of actuators. """
    return 1