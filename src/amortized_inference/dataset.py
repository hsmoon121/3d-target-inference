import os, pickle
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from scipy import stats

from .config import default_config


class SimDataset(object):

    def __init__(
        self,
        total_trial=2**18,
        seed=100,
        prefix="train",
        sim_config=None
    ):
        """
        Initialize the dataset object with a specified number of total instances
        (different user parameter sets, target positions),
        and a simulation configuration.
        """
        self.total_trial = total_trial
        if sim_config is None:
            self.sim_config = deepcopy(default_config["simulator"])
        else:
            self.sim_config = deepcopy(sim_config)
        self.sim_config["seed"] = seed
        
        self.prefix = prefix
        self.name = "dense_" if self.sim_config["dense_grid"] else "wide_"
        if total_trial < 1000:
            self.name += f"{total_trial}"
        else:
            self.name += f"{total_trial//1000}K"

        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/sim_data_{self.prefix}_{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the Raycastingimulator.
        """
        print(f"[ {self.prefix} dataset ] {self.fpath}")
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
            self.norm_info = self.dataset["norm_info"]
        else:
            from .raycasting_simulator import RaycastingSimulator
            self.simulator = RaycastingSimulator(self.sim_config)
            target_arr, params_arr, summ_arr, traj_arr = self.simulator.simulate(
                n_param=self.total_trial,
                verbose=True,
                normalize=True,
            )
            self.norm_info = self.simulator.norm_info
            self.dataset = dict(
                targets=target_arr,
                user_params=params_arr,
                summ_data=summ_arr,
                traj_data=traj_arr,
                norm_info=self.norm_info,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, batch_sz=None, fixed_length=30):
        if batch_sz == None:
            trials = np.arange(self.dataset["targets"].shape[0])
        else:
            trials = np.random.choice(self.dataset["targets"].shape[0], batch_sz, replace=False)
        targets = self.dataset["targets"][trials]
        user_params = self.dataset["user_params"][trials]
        summ_data = self.dataset["summ_data"][trials]
        traj_data = self.dataset["traj_data"][trials]
        self.norm_info = self.dataset["norm_info"]

        traj_data_cursor = list()
        for traj in traj_data:
            traj_data_cursor.append(self.adjust_traj_length(traj[:, 6:9]), fixed_length=fixed_length)
        traj_data_cursor = np.array(traj_data_cursor, dtype=np.float64)
        return (
            np.concatenate((targets, user_params), axis=-1)[:, :3],
            np.concatenate([
                summ_data,
                traj_data_cursor.reshape((-1, 3*fixed_length))
            ], axis=-1)
        )
    
    def adjust_traj_length(self, traj_data, fixed_length=30):
        """
        Adjust the trajectory data to have a fixed length.
        """
        current_length = traj_data.shape[0]
        
        # If the current length is greater than the fixed length, keep only the latter part.
        if current_length > fixed_length:
            return traj_data[-fixed_length:]
        
        # If the current length is less than the fixed length, pad with zeros.
        elif current_length < fixed_length:
            padding_length = fixed_length - current_length
            padding = np.zeros((padding_length, traj_data.shape[1]))
            return np.vstack((padding, traj_data))
        
        # If the current length is equal to the fixed length, return the original data.
        else:
            return traj_data


class SimTrainDataset(SimDataset):
    def __init__(
        self,
        total_trial=2**13,
        sim_config=None
    ):
        super().__init__(
            total_trial=total_trial,
            seed=100,
            prefix="train",
            sim_config=sim_config
        )


class SimValidDataset(SimDataset):
    def __init__(
        self,
        total_trial=500,
        sim_config=None
    ):
        super().__init__(
            total_trial=total_trial,
            seed=121,
            prefix="valid",
            sim_config=sim_config
        )

    def _get_dataset(self):
        """
        Load an existing dataset from file or create a new dataset using the Raycastingimulator.
        """
        print(f"[ {self.prefix} dataset ] {self.fpath}")
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
            self.norm_info = self.dataset["norm_info"]
        else:
            from .raycasting_simulator import RaycastingSimulator
            self.simulator = RaycastingSimulator(self.sim_config)
            
            # target_arr, params_arr, summ_arr, traj_arr
            res_list = self.simulator.simulate(
                n_param=self.total_trial,
                verbose=True,
                normalize=True,
                fix_progress=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            )
            res_dict = [
                dict(
                    targets=res[0],
                    user_params=res[1],
                    summ_data=res[2],
                    traj_data=res[3]
                ) for res in res_list
            ]
            
            self.norm_info = self.simulator.norm_info
            self.dataset = dict(
                res=res_dict,
                norm_info=self.norm_info,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def sample(self, fix_progress, fixed_length=30):
        progress = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert fix_progress in progress

        dataset_progress = self.dataset["res"][progress.index(fix_progress)]
        targets = dataset_progress["targets"]
        user_params = dataset_progress["user_params"]
        summ_data = dataset_progress["summ_data"]
        traj_data = dataset_progress["traj_data"]
        self.norm_info = self.dataset["norm_info"]
        traj_data_cursor = list()
        for traj in traj_data:
            traj_data_cursor.append(self.adjust_traj_length(traj[:, 6:9], fixed_length=fixed_length))
        traj_data_cursor = np.array(traj_data_cursor, dtype=np.float64)
        return (
            np.concatenate((targets, user_params), axis=-1)[:, :3],
            np.concatenate([
                summ_data, 
                traj_data_cursor.reshape((-1, 3*fixed_length))
            ], axis=-1)
        )


class UserDataset():
    def __init__(self):
        self.name = f"user_data"
        self.fpath = os.path.join(
            Path(__file__).parent.parent.parent,
            f"data/{self.name}.pkl"
        )
        self._get_dataset()

    def _get_dataset(self):
        if os.path.exists(self.fpath):
            with open(self.fpath, "rb") as f:
                self.dataset = pickle.load(f)
                self.user_list=self.dataset["user_list"]
        else:
            user_data_path = "./data/study_1/"
            user_data_list = os.listdir(user_data_path)
            user_data_list.sort()
            user_data_list = [user for user in user_data_list if user != ".DS_Store"]

            df = pd.DataFrame()
            for user_i, user in enumerate(user_data_list):
                file_list = os.listdir(os.path.join(user_data_path, user))
                file_list.sort()
                file_list = [file for file in file_list if file != ".DS_Store"]

                for sess_i, file_name in enumerate(file_list):
                    user_data = pd.read_csv(os.path.join(user_data_path, user, file_name))
                    user_data["user_id"] = user_i + 1
                    user_data["session_idx"] = sess_i + 1
                    df = pd.concat([df, user_data], ignore_index=True)

            # make the user_id, session_idx start as the first two columns
            cols = df.columns.tolist()
            cols = cols[-2:] + cols[:-2]
            df = df[cols]

            self.user_list = df["user_id"].unique()
            n_sess = df["session_idx"].max()
            n_block = df["block_idx"].max()

            # Performance (aggregated metrics e.g., completion time & success rate)
            performance_df = pd.DataFrame()
            for user_i in tqdm(self.user_list):
                n_sess = df[
                    (df["user_id"]==user_i)
                ]["session_idx"].max()

                for sess_i in range(n_sess):
                    n_block = df[
                        (df["user_id"]==user_i) & \
                        (df["session_idx"]==sess_i + 1)
                    ]["block_idx"].max()
                        
                    for block_i in range(n_block):
                        n_trial = df[
                            (df["user_id"]==user_i) & \
                            (df["session_idx"]==sess_i + 1)& \
                            (df["block_idx"]==block_i + 1)
                        ]["trial_idx"].max()

                        for trial_i in range(n_trial):
                            df_temp = df[
                                (df["user_id"]==user_i) & \
                                (df["session_idx"]==sess_i + 1) & \
                                (df["block_idx"]==block_i + 1) & \
                                (df["trial_idx"]==trial_i + 1)
                            ]
                            performance_df = pd.concat([performance_df, pd.DataFrame.from_records([{
                                "user_id": user_i,
                                "session_idx": sess_i+1,
                                "block_idx": block_i+1,
                                "trial_idx": trial_i+1,
                                "target_pos_x": df_temp["target_pos_x"].to_numpy()[0].round(3),
                                "target_pos_y": df_temp["target_pos_y"].to_numpy()[0].round(3),
                                "target_pos_z": df_temp["target_pos_z"].to_numpy()[0].round(3),
                                "small_target": df_temp["small_target"].to_numpy()[-1],
                                "small_area": df_temp["small_area"].to_numpy()[-1],
                                "time_elapsed": df_temp["time_elapsed"].to_numpy()[-1],
                                "success": df_temp["success"].to_numpy()[-1],
                                "selection": (df_temp["selection"] == 1).sum(),
                                "start_pos_x": df_temp["position_x"].to_numpy()[0],
                                "start_pos_y": df_temp["position_y"].to_numpy()[0],
                                "start_pos_z": df_temp["position_z"].to_numpy()[0],
                                "end_pos_x": df_temp["position_x"].to_numpy()[-1],
                                "end_pos_y": df_temp["position_y"].to_numpy()[-1],
                                "end_pos_z": df_temp["position_z"].to_numpy()[-1],
                                "start_dir_x": df_temp["direction_x"].to_numpy()[0],
                                "start_dir_y": df_temp["direction_y"].to_numpy()[0],
                                "start_dir_z": df_temp["direction_z"].to_numpy()[0],
                                "end_dir_x": df_temp["direction_x"].to_numpy()[-1],
                                "end_dir_y": df_temp["direction_y"].to_numpy()[-1],
                                "end_dir_z": df_temp["direction_z"].to_numpy()[-1],
                                "eye_height": df_temp["eye_height"].to_numpy()[0],
                            }])], axis=0)
                            
            # Reset index of performance_df
            performance_df = performance_df.reset_index(drop=True)

            # Outlier detection in completion time
            time_z_scores = stats.zscore(performance_df["time_elapsed"])
            abs_time_z_scores = np.abs(time_z_scores)
            time_filtered_entries = (abs_time_z_scores < 3)
            print("Outlier were discarded by completion time: >{:.2f} seconds, {:d} trials ({:.2f} %)".format(
                performance_df[abs_time_z_scores >= 3]["time_elapsed"].min(),
                performance_df[abs_time_z_scores >= 3].shape[0],
                performance_df[abs_time_z_scores >= 3].shape[0] / performance_df.shape[0] * 100.0,
            ))

            # Outlier detection in z-position
            pos_z_scores = stats.zscore(performance_df["end_pos_z"])
            abs_pos_z_scores = np.abs(pos_z_scores)
            pos_filtered_entries = (abs_pos_z_scores < 3)
            print("Outlier were discarded by z-position: >{:.2f}, {:d} trials ({:.2f} %)".format(
                performance_df[abs_pos_z_scores >= 3]["end_pos_z"].min(),
                performance_df[abs_pos_z_scores >= 3].shape[0],
                performance_df[abs_pos_z_scores >= 3].shape[0] / performance_df.shape[0] * 100.0,
            ))

            old_performance_df = performance_df.copy()
            performance_df = performance_df[time_filtered_entries & pos_filtered_entries]

            # Save discarded user_i, sess_i, block_i, trial_i and discard them from the df as well
            banned_df = old_performance_df[(~time_filtered_entries) | (~pos_filtered_entries)][["user_id", "session_idx", "block_idx", "trial_idx"]]
            banned_combinations = set(tuple(x) for x in banned_df[["user_id", "session_idx", "block_idx", "trial_idx"]].values)
            banned_combinations = set(banned_combinations)

            # Filter the DataFrame
            df_filtered = df[~df.apply(lambda row: (row["user_id"], row["session_idx"], row["block_idx"], row["trial_idx"]) in banned_combinations, axis=1)]

            # Average completion time & success rate
            avg_time_elapsed = performance_df["time_elapsed"].mean()
            avg_error_rate = \
                (performance_df["selection"].sum() - performance_df["success"].sum()) / \
                performance_df["selection"].sum()
            print("Average completion time: {:.2f} seconds".format(avg_time_elapsed))
            print("Average error rate: {:.2f}%".format(avg_error_rate*100))

            # Per participant
            avg_time_elapsed_per_user = performance_df.groupby("user_id")["time_elapsed"].mean()
            avg_error_rate_per_user = \
                (performance_df.groupby("user_id")["selection"].sum() - performance_df.groupby("user_id")["success"].sum()) / \
                performance_df.groupby("user_id")["selection"].sum()
            print("Average completion time per user:")
            print(avg_time_elapsed_per_user)
            print("Average error rate per user:")
            print(avg_error_rate_per_user*100)

            self.dataset = dict(
                df=df_filtered,
                performance_df=performance_df,
                user_list=self.user_list,
            )
            with open(self.fpath, "wb") as f:
                pickle.dump(self.dataset, f)

    def set_train_data(
        self, 
        min_window_sz=2, 
        fix_start_idx=True, 
        cross_fold_idx=-1, 
        n_train_user=-1, 
        n_trial_per_user=-1,
        small_target=0,
        dense_grid=1,
    ):
        n_user = len(self.user_list)
        n_fold = 5
        fold_sz = n_user // n_fold
        
        if cross_fold_idx == -1: # all users
            train_users = self.user_list
        else:
            train_users = list()
            for k in range(n_fold):
                if k != cross_fold_idx - 1:
                    train_users.extend(self.user_list[k*fold_sz:(k+1)*fold_sz])

            if n_train_user > 0:
                temp_train_users = train_users + train_users
                train_users = temp_train_users[(cross_fold_idx-1)*fold_sz : (cross_fold_idx-1)*fold_sz + n_train_user]

        res = self.get_trial_data(
            normalize=True,
            user_list=train_users,
            n_trial_per_user=n_trial_per_user,
            small_target=small_target,
            dense_grid=dense_grid,
        )
        
        target_pos_arr, summary_arr, trajectory_arr = list(), list(), list()
        n_trial = res[0].shape[0]
        for trial_i in range(n_trial):
            full_len = res[1][1][trial_i].shape[0]

            # Data augmentation with different window sizes from the same trajectory
            window_size_list = np.arange(min_window_sz, full_len+1)
            for window_size in window_size_list: # range(min_window_sz, full_len+1):
                for start_idx in [0] if fix_start_idx else range(full_len-window_size+1):
                    target_pos_arr.append(res[0][trial_i])
                    summary_arr.append(res[1][0][trial_i])
                    trajectory_arr.append(res[1][1][trial_i][start_idx:start_idx+window_size])

        self.train_dataset = dict(
            target_pos=np.array(target_pos_arr),
            summary=np.array(summary_arr),
            trajectory=np.array(trajectory_arr, dtype=object),
        )
        
    def adjust_traj_length(self, traj_data, fixed_length=30):
        """
        Adjust the trajectory data to have a fixed length.
        """
        current_length = traj_data.shape[0]
        
        # If the current length is greater than the fixed length, keep only the latter part.
        if current_length > fixed_length:
            return traj_data[-fixed_length:]
        
        # If the current length is less than the fixed length, pad with zeros.
        elif current_length < fixed_length:
            padding_length = fixed_length - current_length
            padding = np.zeros((padding_length, traj_data.shape[1]))
            return np.vstack((padding, traj_data))
        
        # If the current length is equal to the fixed length, return the original data.
        else:
            return traj_data

    def sample(
        self, 
        fix_progress, 
        normalize=True, 
        fix_start_idx=True, 
        user_list=None,
        small_target=0,
        dense_grid=1,
    ):
        res = self.get_trial_data(normalize=normalize, user_list=user_list, small_target=small_target, dense_grid=dense_grid)
        
        target_pos_arr, summary_arr, trajectory_arr = list(), list(), list()
        n_trial = res[0].shape[0]
        for trial_i in range(n_trial):
            full_len = res[1][1][trial_i].shape[0]
            window_size = round(full_len * fix_progress)

            for start_idx in [0] if fix_start_idx else range(full_len-window_size+1):
                target_pos_arr.append(res[0][trial_i])
                summary_arr.append(res[1][0][trial_i])
                trajectory_arr.append(res[1][1][trial_i][start_idx:start_idx+window_size]) # full trajectory

        return np.array(target_pos_arr), np.array(summary_arr), np.array(trajectory_arr, dtype=object)
    
    def sample_train_data(self, batch_sz=None, fixed_length=30):
        if batch_sz == None:
            trials = np.arange(self.train_dataset["target_pos"].shape[0])
        else:
            trials = np.random.choice(self.train_dataset["target_pos"].shape[0], batch_sz, replace=False)
        targets = self.train_dataset["target_pos"][trials]
        summ_data = self.train_dataset["summary"][trials]
        traj_data = self.train_dataset["trajectory"][trials]
        traj_data_cursor = list()
        for traj in traj_data:
            traj_data_cursor.append(self.adjust_traj_length(traj[:, 6:9], fixed_length=fixed_length))
        traj_data_cursor = np.array(traj_data_cursor, dtype=np.float64) #, dtype=object)
        return targets, np.concatenate([summ_data, traj_data_cursor.reshape((-1, 3*fixed_length))], axis=-1)

    def sample_valid_data(
        self, 
        fix_progress, 
        normalize=True, 
        fix_start_idx=True, 
        cross_fold_idx=-1,
        small_target=0, 
        dense_grid=1,
        fixed_length=30
    ):
        n_user = len(self.user_list)
        n_fold = 5
        fold_sz = n_user // n_fold
        if cross_fold_idx == -1: # all users
            valid_users = self.user_list
        else:
            valid_users = self.user_list[(cross_fold_idx-1)*fold_sz:(cross_fold_idx)*fold_sz]

        res = self.get_trial_data(normalize=normalize, user_list=valid_users, small_target=small_target, dense_grid=dense_grid)
        
        target_pos_arr, summary_arr, trajectory_arr = list(), list(), list()
        n_trial = res[0].shape[0]
        for trial_i in range(n_trial):
            full_len = res[1][1][trial_i].shape[0]
            window_size = round(full_len * fix_progress)

            for start_idx in [0] if fix_start_idx else range(full_len-window_size+1):
                target_pos_arr.append(res[0][trial_i])
                summary_arr.append(res[1][0][trial_i])

                ### trajctory with cursor only
                trajectory_arr.append(
                    self.adjust_traj_length(res[1][1][trial_i][start_idx:start_idx+window_size, 6:9], fixed_length=fixed_length)
                )
        return (
            np.array(target_pos_arr), 
            np.concatenate([
                np.array(summary_arr),
                np.array(trajectory_arr, dtype=np.float64).reshape((-1, 3*fixed_length))
            ], axis=-1)
        )
    
    def get_trial_data(
        self,
        normalize=False, 
        user_list=None, 
        n_trial_per_user=-1, 
        small_target=0, 
        dense_grid=1
    ):
        if user_list is None:
            user_list = self.dataset["df"]["user_id"].unique()
        n_block = self.dataset["df"]["block_idx"].max()

        target_pos_arr = list()
        summary_arr = list()
        trajectory_arr = list()
        
        assert dense_grid != -1
        if small_target == -1:
            df_cond = self.dataset["df"][self.dataset["df"]["small_area"]==dense_grid]
        else:
            df_cond = self.dataset["df"][(self.dataset["df"]["small_target"]==small_target) \
                                            & (self.dataset["df"]["small_area"]==dense_grid)]
                
        for user_i in user_list:
            
            target_pos_arr_per_user = list()
            summary_arr_per_user = list()
            trajectory_arr_per_user = list()
            
            sess_list = df_cond[df_cond["user_id"]==user_i]["session_idx"].unique()
            for sess_i in sess_list:
                for block_i in range(n_block):
                    trial_list = df_cond[
                        (df_cond["user_id"]==user_i) & \
                        (df_cond["session_idx"]==sess_i)& \
                        (df_cond["block_idx"]==block_i + 1)
                    ]["trial_idx"].unique()

                    for trial_i in trial_list:
                        df_temp = df_cond[
                            (df_cond["user_id"]==user_i) & \
                            (df_cond["session_idx"]==sess_i) & \
                            (df_cond["block_idx"]==block_i + 1) & \
                            (df_cond["trial_idx"]==trial_i)
                        ]
                        target_pos = np.array([
                            df_temp["target_pos_z"].to_numpy()[0],
                            df_temp["target_pos_x"].to_numpy()[0],
                            df_temp["target_pos_y"].to_numpy()[0],
                        ])

                        target_radius, obj_positions, _ = \
                            self.get_task_info(df_temp["small_area"].to_numpy()[0], df_temp["small_target"].to_numpy()[0])
                        summary = np.concatenate([
                            np.array([target_radius,]),
                            obj_positions.flatten(),
                        ])

                        ray_trajectory = np.array([
                            df_temp["position_z"].to_numpy(),
                            df_temp["position_x"].to_numpy(),
                            df_temp["position_y"].to_numpy(),
                            df_temp["direction_z"].to_numpy(),
                            df_temp["direction_x"].to_numpy(),
                            df_temp["direction_y"].to_numpy(),
                        ]).T
                        projected_pos = self.projection_of_ray(
                            ray_trajectory[:, :3],
                            ray_trajectory[:, 3:],
                        )
                        obj_distances = self.distance_to_objects(
                            ray_trajectory[:, :3],
                            ray_trajectory[:, 3:],
                            obj_positions + np.array([0.0, 0.0, 1.2]),
                        )
                        trajectory = np.concatenate([
                            ray_trajectory,
                            projected_pos,
                            obj_distances,
                        ], axis=-1)

                        if normalize:
                            target_pos, summary, trajectory = self.normalize(target_pos, summary, trajectory)

                        target_pos_arr_per_user.append(target_pos)
                        summary_arr_per_user.append(summary)
                        trajectory_arr_per_user.append(trajectory)

            if n_trial_per_user > len(target_pos_arr_per_user) or n_trial_per_user == -1:
                random_indices = np.arange(len(target_pos_arr_per_user))
            else:
                random_indices = np.random.choice(len(target_pos_arr_per_user), n_trial_per_user, replace=False)

            target_pos_arr.extend(np.array(target_pos_arr_per_user)[random_indices])
            summary_arr.extend(np.array(summary_arr_per_user)[random_indices])
            trajectory_arr.extend(np.array(trajectory_arr_per_user, dtype=object)[random_indices])
        
        target_pos_arr = np.array(target_pos_arr)
        summary_arr= np.array(summary_arr)
        trajectory_arr = np.array(trajectory_arr, dtype=object)
        return target_pos_arr, (summary_arr, trajectory_arr)
  
    def get_task_info(self, dense_grid, small_target):
        radius = 0.03 if small_target else 0.05
        n_row, n_column = [7, 7] if dense_grid else [7, 9]
        spacing = 1.44 if dense_grid else 6.0
        target_dist = 5.0
        
        n_obj = n_row * n_column
        obj_positions = list()

        for obj_i in range(1, n_obj+1):
            row = (obj_i-1) // n_column
            column = (obj_i-1) % n_column
            phi = -(column - (n_column-1)/2) * spacing * math.pi / 180.0 # azimuth angle
            theta = -(row - (n_row-1)/2) * spacing * math.pi / 180.0 # elevation angle

            depth = target_dist * math.cos(theta) * math.cos(phi)
            x = target_dist * math.cos(theta) * math.sin(phi)
            y = target_dist * math.sin(theta)
            obj_pos = np.array([depth, x, y])

            # candidates should not be on the outer-most layer
            if row * (row-n_row+1) * column * (column-n_column+1) != 0:
                # candidates also should not be on the center & inner-most layer in larger grid
                if dense_grid or abs(row - (n_row-1)/2) > 1 or abs(column - (n_column-1)/2) > 1:
                    obj_positions.append(obj_pos * np.array([1, -1, 1]))

        self.norm_info = dict(
            max_target_pos = np.array([5.0, 1.6, 1.6]), # (depth, x, y) orders
            min_target_pos = np.array([4.5, -1.6, -1.6]),
            max_summary = np.array([0.05] + [5.0, 1.6, 1.6] * len(obj_positions)),
            min_summary = np.array([0.0] + [4.5, -1.6, -1.6] * len(obj_positions)),
            max_trajectory = np.array([1.0, 0.30, 1.2, 1.0, 0.5, 0.5, 5.0, 1.6, 1.6] + [2.0,] * len(obj_positions)),
            min_trajectory = np.array([0.0, -0.10, 0.6, 0.8, -0.5, -0.5, 4.5, -1.6, 0] + [0.0,] * len(obj_positions)),
        )
        return radius, np.array(obj_positions), self.norm_info
  
    def projection_of_ray(self, pos_arr, dir_arr, center=np.array([0.0, 0.0, 1.2]), target_dist=5.0):
        # Track the position of the projected ray on the same-distance surface
        # Ray P = pos + dir * t
        # Sphere: ||P - O|| = target_dist

        # Quadratic formula coefficients
        A = np.sum(dir_arr**2, axis=1)
        B = 2 * np.sum((pos_arr-center)*dir_arr, axis=1)
        C = np.sum((pos_arr-center)**2, axis=1) - target_dist**2

        # Discriminant
        D = B**2 - 4*A*C
        t = (-B + np.sqrt(D)) / (2*A)
        return pos_arr + t.reshape((-1, 1)) * dir_arr
    
    def distance_to_objects(self, pos_arr, dir_arr, obj_pos_arr):
        # Normalize the direction vectors
        ray_dir_arr = dir_arr / np.linalg.norm(dir_arr, axis=-1, keepdims=True)

        # Reshape the arrays for broadcasting:
        # ray_pos_arr_b and ray_dir_arr_b: (num_steps, 1, 3)
        # obj_pos_arr_b: (1, num_objects, 3)
        ray_pos_arr_b = np.expand_dims(pos_arr, axis=1)
        ray_dir_arr_b = np.expand_dims(ray_dir_arr, axis=1)
        obj_pos_arr_b = np.expand_dims(obj_pos_arr, axis=0)

        diffs = obj_pos_arr_b - ray_pos_arr_b
        dots = np.sum(diffs * ray_dir_arr_b, axis=-1)
        projs = np.sum(diffs * ray_dir_arr_b, axis=-1, keepdims=True) * ray_dir_arr_b

        orths = diffs - projs
        distances = np.linalg.norm(orths, axis=-1)
        distances[dots < 0] = np.linalg.norm(diffs, axis=-1)[dots < 0]
        return distances
  
    def normalize(self, target_pos, summary, trajectory):
        return (
        (target_pos - self.norm_info["min_target_pos"]) / \
            (self.norm_info["max_target_pos"] - self.norm_info["min_target_pos"]) * 2 - 1,
        (summary - self.norm_info["min_summary"]) / \
            (self.norm_info["max_summary"] - self.norm_info["min_summary"]) * 2 - 1,
        (trajectory - self.norm_info["min_trajectory"]) / \
            (self.norm_info["max_trajectory"] - self.norm_info["min_trajectory"]) * 2 - 1,
        )