default_config = dict(
    name = "sim_dense",
    learning_rate = 0.0001,
    lr_gamma = 0.9,
    clipping = 0.5,

    cross_fold_idx = -1, # for supervised learning
    n_train_user = -1, # for supervised learning
    n_trial_per_user = -1, # for supervised learning

    dense_grid = 1,
    small_target = -1,
    fixed_length = 30,
    amortizer = dict(
        device = None,
        trial_encoder_type = None,
        encoder = dict(
            # rather than using traj_encoder (encoder for time-series data), 
            # we used the last 30 steps of end effector positions as part of our inputs to MLP
            traj_sz = 0,
            stat_sz = 1 + 3*25 + 3*30,
            batch_norm = True,
            traj_encoder_type = "conv_rnn", # either "conv_rnn" or"transformer"
            conv1d = [],
            rnn = dict( # unused as traj_sz=0
                type = "LSTM",
                bidirectional = False,
                dropout = 0.2,
                feat_sz = 32,
                depth = 2,
            ),
            mlp = dict(
                feat_sz = 128,
                depth = 2,
                out_sz = 128,
            ),
        ),
        invertible = dict(
            param_sz = 3,
            n_block = 5,
            act_norm = False,
            invert_conv = False,
            batch_norm = False,
            block = dict(
                permutation = True,
                head_depth = 2,
                cond_sz = 128,
                head_sz = 32,
                feat_sz = 32,
                depth = 2,
            )
        )
    ),
    simulator = dict(
        seed = None,
        dense_grid = True,
        small_target = -1,
        config_path = "./configs/train.yaml",
        targeted_params = [
            "target_pos_z",
            "target_pos_x",
            "target_pos_y",
        ],
        param_symbol = [
            "$pos_z$",
            "$pos_x$",
            "$pos_y$",
        ],
    ),
)