from scipy.spatial.transform import Rotation
import pathlib
import os
from datetime import datetime
import sys
import select
import scipy.ndimage
import numpy as np
from distutils.dir_util import copy_tree
import re
from ruamel.yaml import YAML

from .transformations import transformation_matrix


def parent_path(file):
  return pathlib.Path(file).parent.absolute()

def project_path():
  return pathlib.Path(__file__).parent.parent.parent.absolute()

def output_path():
  return os.path.join(project_path(), "simulators")

def strtime():
  return datetime.utcfromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%dT%H-%M-%SZ')

def timeout_input(prompt, timeout=30, default=""):
  print(prompt, end='\n>> ', flush=True)
  inputs, outputs, errors = select.select([sys.stdin], [], [], timeout)
  print()
  return sys.stdin.readline().strip() if inputs else default

# Numerically stable sigmoid
def sigmoid(x):
  return np.exp(-np.logaddexp(0, -x))

def is_suitable_package_name(name):
  match = re.match("^[a-z0-9_]*$", name)
  return match is not None and name[0].isalpha()

def parse_yaml(yaml_file):
  yaml = YAML()
  with open(yaml_file, 'r') as stream:
    parsed = yaml.load(stream)
  return parsed

def write_yaml(data, file):
  yaml = YAML()
  with open(file, "w") as stream:
    yaml.dump(data, stream)

def img_history(imgs, k=0.9):

  # Make sure intensities are properly normalised
  N = len(imgs)

  img = np.zeros_like(imgs[0], dtype=np.float)
  norm = 0

  for i in range(N):
    coeff = np.exp(-((N-1)-i)*k)
    img += coeff * imgs[i]
    norm += coeff

  return img / (255*norm)

def grab_pip_image(simulator):
  # Grab an image from both 'for_testing' camera and 'oculomotor' camera, and display them 'picture-in-picture'

  # Grab images
  img, _ = simulator._camera.render()

  ocular_img = None
  for module in simulator.perception.perception_modules:
    if module.modality == "vision":
      # TODO would be better to have a class function that returns "human-viewable" rendering of the observation;
      #  e.g. in case the vision model has two cameras, or returns a combination of rgb + depth images etc.
      ocular_img = module.render()
      # ocular_img, depth = module._camera.render()

  if ocular_img is not None:

    # Resample
    resample_factor = 2
    resample_height = ocular_img.shape[0]*resample_factor
    resample_width = ocular_img.shape[1]*resample_factor
    resampled_img = np.zeros((resample_height, resample_width, 3), dtype=np.uint8)
    for channel in range(3):
      resampled_img[:, :, channel] = scipy.ndimage.zoom(ocular_img[:, :, channel], resample_factor, order=0)
      # resampled_img[:, :, channel] = scipy.ndimage.zoom(depth[:, :]*255, resample_factor, order=0)
      

    # Embed ocular image into free image
    i = simulator._camera.height - resample_height
    j = simulator._camera.width - resample_width
    img[i:, j:] = resampled_img

  if hasattr(simulator.task, "_count_for_click"): # whether it is a raycasting task
    # Flag for click decision / click execution
    if simulator.task._count_for_click > 0:
      flag_sz = 40
      color_start = np.ones((flag_sz, flag_sz, 3)) * np.array([255, 63, 63])
      color_end = np.ones((flag_sz, flag_sz, 3)) * np.array([63, 255, 63])
      img[:flag_sz, :flag_sz] = color_start + (color_end-color_start) * (simulator.task._count_for_click/simulator.task._click_delay)

    # Flag for selection success / failure
    if simulator.task._info["require_start"] and simulator.task._trial_idx > 0:
      flag_sz = 40
      offset = 5
      if simulator.task._info["hit_at_last_trial"]:
        color = np.ones((flag_sz, flag_sz, 3)) * np.array([63, 255, 63])
      else:
        color = np.ones((flag_sz, flag_sz, 3)) * np.array([255, 63, 63])
      img[flag_sz+offset:flag_sz*2+offset, :flag_sz] = color

  return img

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def initialise_pos_and_quat(model, data, aux_body, relpose, body):
  """ Initialise pos and quat of body according to the relpose wrt to aux_body"""
  T1 = transformation_matrix(pos=data.body(aux_body).xpos, quat=data.body(aux_body).xquat)
  T2 = transformation_matrix(pos=relpose[:3], quat=relpose[3:])
  T = np.matmul(T1, np.linalg.inv(T2))
  model.body(body).pos = T[:3, 3]
  model.body(body).quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)