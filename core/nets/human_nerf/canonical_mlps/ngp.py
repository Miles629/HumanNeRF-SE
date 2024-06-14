import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import tinycudann as tcnn

EPS = 1e-3

class TruncExp(Function):
    # @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x.clamp(max=15, min=-15))

    # @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(max=15, min=-15))


class NeRFNGPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder =  tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=16,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.5,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        self.color_net = tcnn.Network(
            n_input_dims=16,
            n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            },
        )
        # self.sigma_activ = lambda x: torch.relu(x).float()
        # self.sigma_activ = TruncExp.apply

        self.register_buffer("center", torch.FloatTensor([0, -0.3, 0]))
        self.register_buffer("scale", torch.FloatTensor([2.5, 2.5, 2.5]))
        
        self.use_viewdir = False
        self.cond_dim = 0

    def initialize(self, bbox):
        if hasattr(self, "bbox"):
            return
        c = (bbox[0] + bbox[1]) / 2
        s = (bbox[1] - bbox[0])
        self.center = c
        self.scale = s
        self.bbox = bbox

    def forward(self, x, d=None, cond=None):
        # normalize pts and view_dir to [0, 1]
        x = (x - self.center) / self.scale + 0.5
        assert x.min() >= -EPS and x.max() < 1 + EPS
        x = x.clamp(min=0, max=1)
        out = self.encoder(x)
        # x = self.density_net(x)

        # sigma = out[..., 0]
        # color = self.color_net(out[..., 1:]).float()
        out = self.color_net(out)
        color = out[...,:3]
        sigma = out[...,3]

        # sigma = self.sigma_activ(sigma)
        return color.float(), sigma.float()
