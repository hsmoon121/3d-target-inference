from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .nets import AmortizerForSummaryData
from .dataset import UserDataset
from .utils import CosAnnealWR
from .config import default_config


class RaycastingSupervisedTrainer(BaseTrainer):
    """
    Inference network trainer for user data-based supervised learning
    """
    def __init__(self, config=None):
        self.config = deepcopy(default_config) if config is None else config
        super().__init__(name=self.config["name"], task_name="raycasting")

        # Initialize the amortizer, simulator, and datasets
        self.amortizer = AmortizerForSummaryData(config=self.config["amortizer"])
        self.dataset = UserDataset()

        self.targeted_params = self.config["simulator"]["targeted_params"]
        self.param_symbol = np.array(self.config["simulator"]["param_symbol"])
        self.cross_fold_idx = self.config["cross_fold_idx"]
        self.n_train_user = self.config["n_train_user"]
        self.n_trial_per_user = self.config["n_trial_per_user"]
        self.dense_grid = self.config["dense_grid"]
        self.small_target = self.config["small_target"]
        self.fixed_length = self.config["fixed_length"]

        # Initialize the optimizer and scheduler
        self.lr = self.config["learning_rate"]
        self.lr_gamma = self.config["lr_gamma"]
        self.clipping = self.config["clipping"]
        self.optimizer = torch.optim.Adam(self.amortizer.parameters(), lr=1e-9)
        self.scheduler = CosAnnealWR(self.optimizer, T_0=10, T_mult=1, eta_max=self.lr, T_up=1, gamma=self.lr_gamma)

    def train(
        self,
        n_iter=100,
        step_per_iter=2000,
        batch_sz=2048,
        train_mode="replay",
        capacity=100000,
        board=True
    ):
        super().train(
            n_iter=n_iter,
            step_per_iter=step_per_iter,
            batch_sz=batch_sz,
            train_mode=train_mode,
            capacity=capacity,
            board=board,
        )

    def _set_train_mode(self, train_mode, capacity):
        """
        Set the training mode for the trainer ("offline")
        """
        if train_mode == "offline":
            self.dataset = UserDataset()
            self.dataset.set_train_data(
                cross_fold_idx=self.cross_fold_idx,
                n_train_user=self.n_train_user,
                n_trial_per_user=self.n_trial_per_user,
                dense_grid=self.dense_grid,
                small_target=self.small_target,
            )
            _, _, self.norm_info = self.dataset.get_task_info(dense_grid=self.dense_grid, small_target=self.small_target)
        else:
            raise RuntimeError(f"Wrong training type: {train_mode}")
    
    def _get_offline_batch(self, batch_sz):
        return self.dataset.sample_train_data(batch_sz=batch_sz, fixed_length=self.fixed_length)

    def valid(
        self,
        n_sample=100,
        infer_type="mode",
        plot=True,
    ):
        self.amortizer.eval()
        valid_res = dict()
        self.fig_path = f"{self.result_path}/{self.name}/iter{self.iter:03d}/"
        os.makedirs(f"{self.result_path}/{self.name}", exist_ok=True)
        os.makedirs(self.fig_path, exist_ok=True)

        # Infer the posterior of the validation dataset
        self.inference_on_data(valid_res, self.dataset, "user", n_sample, infer_type, plot)
        return valid_res
    
    def inference_on_data(self, res, dataset, prefix="sim", n_sample=100, infer_type="mode", plot=True, stride=4):
        progress_list = list(np.arange(0.1, 1.1, 0.1))
        acc_per_progress, prob_per_progress = list(), list()

        for progress in tqdm(progress_list):
            gt_targets, gt_summ_data = dataset.sample_valid_data(
                fix_progress=progress,
                cross_fold_idx=self.cross_fold_idx,
                dense_grid=self.dense_grid,
                small_target=self.small_target,
                fixed_length=self.fixed_length
            )
            n_param = gt_targets.shape[0]

            # normalized positions of target candidates
            target_info = gt_summ_data[0, 1:].reshape((-1, 3))

            map_arr, data_sz_arr = list(), list()
            acc_arr, prob_arr = list(), list()

            for param_i in range(0, n_param, stride):
                summ_i = gt_summ_data[param_i]
                data_sz_arr.append(1)

                # Unsqueeze (as 1-size batch)
                summ_i = np.expand_dims(summ_i, axis=0)

                # Get MAP fit
                map_fit = self.amortizer.infer(summ_i, n_sample=n_sample, type=infer_type)
                map_arr.append(self._clip_params(map_fit))

                # Fix the other fitted user parameters
                other_params = map_fit[3:]
                other_params = np.expand_dims(other_params, axis=0)
                other_params = np.repeat(other_params, target_info.shape[0], axis=0)
                candidates = np.concatenate([target_info, other_params], axis=-1)

                # Get the probability of the other candidates
                summs = np.repeat(summ_i, candidates.shape[0], axis=0)
                densities = self.amortizer.pdf(candidates, summs) + 1e-30
                probabilities = densities / np.nansum(densities)
                max_target_i = np.argmax(probabilities)

                acc_arr.append(np.allclose(candidates[max_target_i, :3], gt_targets[param_i, :3], atol=1e-3))
                prob_arr.append(probabilities[max_target_i])
                
            map_arr = np.array(map_arr)
            data_sz_arr = np.array(data_sz_arr)
            acc_arr = np.array(acc_arr)
            prob_arr = np.array(prob_arr)
            acc_per_progress.append(acc_arr)
            prob_per_progress.append(prob_arr)

            if progress in [0.2, 0.6, 1.0]:
                res[f"inference_accuracy/{prefix}_{progress*100:.0f}"] = acc_arr.mean()
                res[f"inference_confidence/{prefix}_{progress*100:.0f}"] = prob_arr.mean()
                self.parameter_recovery(res, gt_targets[:n_param:stride], map_arr, prefix=f"{prefix}_{progress*100:.0f}", plot=plot)

        if plot:
            self.plot_acc_prob_per_progress(np.array(acc_per_progress), np.array(prob_per_progress))
        
    def parameter_recovery(self, res, y_true, y_pred, prefix, plot=True):
        # Plot parameter recovery
        n_param = min(y_true.shape[-1], y_pred.shape[-1])
        for i, l in enumerate(self.targeted_params[:n_param]):
            yi_true = y_true[:, i]
            yi_pred = y_pred[:, i]

            if np.std(yi_true) == 0:
                # In case there are no ground-truth values (i.e., all values are 0)
                pass
            else:
                y_fit = np.polyfit(yi_true, yi_pred, 1)
                y_func = np.poly1d(y_fit)
                r_squared = r2_score(yi_pred, y_func(yi_true))

                res[f"parameter_recovery/{prefix}_r2_" + l] = r_squared
                if plot:
                    self._plot_parameter_recovery(
                        yi_true,
                        yi_pred,
                        y_fit,
                        r_squared,
                        fname=f"{prefix}_r2_" + l,
                        param_label=self.param_symbol[i],
                    )

    def plot_acc_prob_per_progress(self, acc_per_progress, prob_per_progress):
        # Plot inference accuracy per fraction of trajectory
        # Error bar with 95% confidence interval
        fig = plt.figure(figsize=(10, 6))
        x = np.arange(0.1, 1.1, 0.1)
        y = acc_per_progress.mean(axis=-1)
        y_ci = 1.96 * acc_per_progress.std(axis=-1) / np.sqrt(acc_per_progress.shape[-1])

        plt.plot(x, y, marker="o", color="blue", label="Accuracy")
        plt.fill_between(x, y-y_ci, y+y_ci, color="blue", alpha=0.3, label="95% CI")
        plt.ylabel("Inference accuracy")
        plt.xlabel("Fraction of trajectory observed")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, f"accuracy_over_fraction.pdf"), dpi=300)
        plt.close(fig)

        # Plot inference confidence per fraction of trajectory
        fig = plt.figure(figsize=(10, 6))
        x = np.arange(0.1, 1.1, 0.1)
        y = prob_per_progress.mean(axis=-1)
        y_ci = 1.96 * prob_per_progress.std(axis=-1) / np.sqrt(prob_per_progress.shape[-1])
        
        plt.plot(x, y, marker="o", color="blue", label="Accuracy")
        plt.fill_between(x, y-y_ci, y+y_ci, color="blue", alpha=0.3, label="95% CI")
        plt.ylabel("Inference confidence (inferred probability)")
        plt.xlabel("Fraction of trajectory observed")
        plt.ylim([0.0, 1.0])
        plt.tight_layout()
        plt.savefig(os.path.join(self.fig_path, f"confidence_over_fraction.pdf"), dpi=300)
        plt.close(fig)

    def plot_sample_sequence(self, gt_targets, summ, traj):
        # Plot sample sequence of inferred posteriors
        sample_len = traj.shape[1]
        for t in range(2, sample_len, 2):

            map_fit, samples = self.amortizer.infer(
                summ,
                traj[:, :t],
                n_sample=100, 
                type="mode",
                return_samples=True
            )            
            self._pair_plot(
                samples,
                self.param_symbol,
                limits=np.array([[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5]]),
                gt_params=np.concatenate([gt_targets[:3], map_fit[3:]]),
                fname=f"user_sample_posterior_{t}"
            )
        
    def _clip_params(self, params):
        return np.clip(
            params,
            -1, # self.norm_info["min_target_pos"],
            1, #self.norm_info["max_target_pos"],
        )