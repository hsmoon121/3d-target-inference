from copy import deepcopy

from src.amortized_inference.raycasting_trainer import RaycastingTrainer
from src.amortized_inference.raycasting_supervised_trainer import RaycastingSupervisedTrainer
from src.amortized_inference.config import default_config

def main(args):

    if args.supervised:
        config = deepcopy(default_config)
        config["cross_fold_idx"] = args.k_fold
        config["n_train_user"] = args.n_user
        config["n_trial_per_user"] = args.n_trial_per_user
        config["dense_grid"] = args.dense_grid
        config["small_target"] = args.small_target
        config["amortizer"]["encoder"]["stat_sz"] = (1 + 3*25 + 3*30) if args.dense_grid == 1 else (1 + 3*26 + 3*30)

        name = f"human"
        if args.dense_grid != -1:
            name += f"_dense" if args.dense_grid == 1 else f"_wide"
        if args.small_target != -1:
            name += f"_small" if args.small_target == 1 else f"_large"

        name += "_all" if args.k_fold == -1 else f"_{args.k_fold}"
        if config["n_train_user"] != -1:
            name += f"_{args.n_user}user"
        if config["n_trial_per_user"] != -1:
            name += f"_{args.n_trial_per_user}trial"
        config["name"] = name

        raycasting_trainer = RaycastingSupervisedTrainer(config=config)
    
    else:
        config = deepcopy(default_config)
        config["dense_grid"] = args.dense_grid
        config["small_target"] = args.small_target
        config["fixed_length"] = args.fixed_length
        config["amortizer"]["encoder"]["stat_sz"] = (1 + 3*25 + 3*30) if args.dense_grid == 1 else (1 + 3*26 + 3*30)
        config["simulator"]["dense_grid"] = args.dense_grid
        config["simulator"]["small_target"] = args.small_target

        name = f"sim"
        if args.dense_grid != -1:
            name += f"_dense" if args.dense_grid == 1 else f"_wide"
        if args.small_target != -1:
            name += f"_small" if args.small_target == 1 else f"_large"
        config["name"] = name

        raycasting_trainer = RaycastingTrainer(config=config)

    raycasting_trainer.load()
    raycasting_trainer.train(train_mode="replay")

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_fold", type=int, default=1)
    parser.add_argument("--n_user", type=int, default=-1)
    parser.add_argument("--n_trial_per_user", type=int, default=-1)
    parser.add_argument("--dense_grid", type=int, default=1)
    parser.add_argument("--small_target", type=int, default=-1) # 1: Small / 0: Large / -1: Both
    parser.add_argument("--supervised", action="store_true")
    args = parser.parse_args()
    main(args)