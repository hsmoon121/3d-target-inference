import numpy as np
import torch
from scipy.stats import multivariate_normal

from .base import NetFrame
from .encoder import EncoderNet
from .invertible import InvertibleNet
from .utils import get_auto_device


class AmortizerFrame(NetFrame):
    def __init__(self, config):
        super().__init__(config)

    def _set_device(self, config):
        """
        Set device for the model. 
        If device is not specified in the config, it will be automatically set.

        config (dict): Configuration dictionary for the amortizer model.
        """
        if "device" not in config:
            self.given_device = get_auto_device()
        elif config["device"] is None:
            self.given_device = get_auto_device()
        else:
            self.given_device = torch.device(config["device"])
        self.to(self.given_device)


    def infer(self, inputs, n_sample=100, infer_type="mode", return_samples=False):
        """
        Infer the posterior distribution of the parameters given the static and trajectory data.

        inputs (dict)
        n_sample (int, optional): Number of posterior samples, default is 100.
        infer_type (str, optional): Type of inference from distribution ("mode", "mean", "median"), default is "mode".
        return_samples (bool, optional): Whether to return posterior samples, default is False.
        ---
        outputs (tuple): Tuple containing inferred parameters (torch.Tensor) and posterior samples (torch.Tensor).
        """
        n_param = self.invertible_net.n_param
        cond = self.encoder_net(inputs)
        param_sampled, logpdf = self._sample(inputs, n_sample, return_logpdf=True)

        if infer_type == "mode": # Maximum a Posteriori (MAP) estimation
            if cond.shape[0] == 1:
                res = param_sampled[np.argmax(logpdf, axis=-1)]
            else:
                batch_indices = np.arange(param_sampled.shape[0])
                res = param_sampled[batch_indices, np.argmax(logpdf, axis=-1)]
        
        elif infer_type == "mean":
            res = np.mean(param_sampled, axis=-1)

        elif infer_type == "median":
            res = np.median(param_sampled, axis=-1)

        else:
            raise Exception(f"inappropriate type: {infer_type}")

        if return_samples:
            return res, param_sampled
        return res


class AmortizerForSummaryData(AmortizerFrame):
    def __init__(self, config=dict()):
        """
        Amortizer model for summary data (i.e., fixed-size data per inference)

        config (dict): Configuration dictionary for the amortizer model.
        """
        super().__init__(config)
        self.encoder_net = EncoderNet(config["encoder"])
        self.invertible_net = InvertibleNet(config["invertible"])
        self._set_device(config)

    def forward(self, params, stat_data):
        """
        params (ndarray): array of parameters with a shape (n_batch, n_param).
        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        ---
        outputs (list): List of outputs from the invertible network.
        """
        assert not self.encoder_net.series_data
        params = torch.FloatTensor(params).to(self.device)
        cond = self.encoder_net(stat_data)
        out = self.invertible_net(params, cond)
        return out

    def _sample(self, stat_data, n_sample=100, return_logpdf=False):
        """
        Sample from the posterior distribution of the parameters given the static and trajectory data.

        stat_data (ndarray): array of static data with a shape (n_batch, stat_sz).
        traj_data (list, optional): List of trajectory data, each item should have a shape (traj_length, traj_sz).
        n_sample (int, optional): Number of posterior samples, default is 100.
        ---
        outputs (tuple): Tuple containing posterior samples (ndarray) and log determinant of Jacobian (ndarray).
        """
        assert not self.encoder_net.series_data
        cond = self.encoder_net(stat_data)
        return self.invertible_net.sample(cond, n_sample, return_logpdf=return_logpdf)
    
    def pdf(self, params, stat_data):
        assert not self.encoder_net.series_data
        n_param = self.invertible_net.n_param
        params = torch.FloatTensor(params).to(self.device)
        cond = self.encoder_net(stat_data)
        out = self.invertible_net(params, cond)
        z, log_det_J = [o.cpu().detach().numpy() for o in out]
        base_log_prob = multivariate_normal.logpdf(z, mean=np.zeros(n_param), cov=np.eye(n_param))
        return np.exp(base_log_prob + log_det_J)