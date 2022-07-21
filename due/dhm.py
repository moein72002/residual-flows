import math
import torch
import numpy as np
import torch.nn as nn


from residual_flows.lib.resflow import ResidualFlow
from residual_flows.lib.layers import LogitTransform


class DHM(nn.Module):
    def __init__(self, feature_extractor, hparams,):
        super().__init__()
        self.feature_extractor = feature_extractor
        """For flow-based model, we use the standard setup
        of passing the data through a logit transform [6], followed by 10 
        residual blocks.
        We use activation normalization [13] before and after every residual 
        block. Each
        residual connection consists of 6 layers (i.e., LipSwish [4] → 
        InducedNormLinear
        → LipSwish → InducedNormLinear → LipSwish → InducedNormLinear) with
        hidden dimensions of 256 (the first 6 blocks) and 128 (the next 4 
        blocks) [20]."""

        self.batch_size = hparams.batch_size
        self.input_size = (hparams.batch_size, 1, 32, 20)
        init_layer = LogitTransform(0.05)
        n_classes = 10 if hparams.dataset == 'CIFAR10' else 100
        self.normalizing_flow = ResidualFlow(
            self.input_size,
            n_blocks=list(map(int, '16-16-16'.split('-'))),
            intermediate_dim=640,  # MAYBE 512
            factor_out=False,
            quadratic=False,
            init_layer=init_layer,
            actnorm=True,
            fc_actnorm=False,
            batchnorm=False,
            dropout=0.,
            fc=False,
            coeff=hparams.coeff,
            vnorms='2222',
            n_lipschitz_iters=None,
            sn_atol=1e-3,
            sn_rtol=1e-3,
            n_power_series=None,
            n_dist='poisson',
            n_samples=1,
            kernels='3-1-3',
            activation_fn='swish',
            fc_end=False,
            fc_idim=128,
            n_exact_terms=2,
            preact=True,
            neumann_grad=True,
            grad_in_forward=True,
            first_resblock=True,
            learn_p=False,
            classification=False,
            classification_hdim=640,
            n_classes=10 if hparams.dataset == 'CIFAR10' else 100,
            block_type='resblock',
        )
        self.fully_connected = nn.Linear(640, n_classes)

    def forward(self, x):
        h_features = self.feature_extractor(x)
        z_features = self.normalizing_flow(h_features.reshape(
            self.input_size), 0)
        logits = self.fully_connected(h_features)
        return z_features, logits


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def compute_nf_loss(z_features, logits_tensor, beta=1.0):
    z, delta_logp = z_features
    nvals = 256
    imagesize = 32
    im_dim = 3
    padding = 0

    # log p(z)
    logpz = standard_normal_logprob(z).view(z.size(0), -1).sum(1, keepdim=True)
    # log p(x)
    logpx = logpz - beta * delta_logp - np.log(nvals) * (
            imagesize * imagesize * (im_dim + padding)
    )
    bits_per_dim = -torch.mean(logpx) / (
                imagesize * imagesize * im_dim) / np.log(2)

    logpz = torch.mean(logpz).detach()
    delta_logp = torch.mean(-delta_logp).detach()
    return bits_per_dim, logits_tensor, logpz, delta_logp
