import math
import torch
import numpy as np
import torch.nn as nn


class Reshaper(nn.Module):
    def __init__(self, image_shape):
        super(Reshaper, self).__init__()
        self.image_shape = image_shape

    def forward(self, x):
        return x.view(-1, *self.image_shape)


class DHM(nn.Module):
    def __init__(self, feature_extractor, normalizing_flow, n_classes, nf_input_size):
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
        self.nf_input_size = nf_input_size
        self.reshaper = Reshaper(self.nf_input_size[1:])
        self.normalizing_flow = normalizing_flow
        self.fully_connected = nn.Linear(640, n_classes)

    def forward(self, x, logpx=None, inverse=False, classify=False):
        h_features = self.feature_extractor(x)
        h_features_reshaped = self.reshaper(h_features)
        z_features = self.normalizing_flow(h_features_reshaped, logpx,
                                           inverse, classify)
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
