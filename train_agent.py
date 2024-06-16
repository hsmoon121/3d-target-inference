import wandb
from wandb.integration.sb3 import WandbCallback

from src.simulator import Simulator
from src.utils.functions import output_path


def main(args):

  # Get config file path
  config_file_path = args.config_path
  eval_config_file_path = args.eval_config_path

  # Build the simulator
  simulator_folder = Simulator.build(config_file_path)
  _ = Simulator.build(eval_config_file_path)

  # Initialise
  simulator = Simulator.get(simulator_folder)

  # Get the config
  config = simulator.config

  # Get simulator name
  model_name = config.get("model_name", None)

  # Get project name
  project = config.get("project", "uitb")

  # Initialise wandb
  if args.use_wandb:
    run = wandb.init(project=project, name=model_name, config=config, sync_tensorboard=True, save_code=True, dir=output_path())

  # Initialise RL model
  rl_cls = simulator.get_class("rl", config["rl"]["algorithm"])
  rl_model = rl_cls(simulator, model_name=model_name)

  # Start the training
  rl_model.load()

  if args.use_wandb:
    rl_model.learn(WandbCallback(verbose=2))
    run.finish()
  else:
    rl_model.learn()


if __name__ == "__main__":

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", type=str, default="./configs/train.yaml")
  parser.add_argument("--eval_config_path", type=str, default="./configs/eval.yaml")
  parser.add_argument("--use_wandb", type=int, default=0)
  args = parser.parse_args()

  main(args)