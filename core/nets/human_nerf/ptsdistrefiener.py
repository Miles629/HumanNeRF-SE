import torch.nn as nn

from core.utils.network_util import initseq, RodriguesModule

from configs import cfg

class PtsRefiner(nn.Module):
    def __init__(self,
                 embedding_size=5,
#                  mlp_width=256,
                 mlp_depth=4,
                 **_):
        super(PtsRefiner, self).__init__()
        
        mlp_width = 64
        width2 = 32
        width4 = 16

#         print(width2.dtype,mlp_width.dtype)
        block_mlps = [nn.Linear(embedding_size, width4), nn.ReLU()]

#         for _ in range(0, mlp_depth-1):
        block_mlps += [nn.Linear(width4, width2), nn.ReLU()]
        block_mlps += [nn.Linear(width2, mlp_width), nn.ReLU()]
        block_mlps += [nn.Linear(mlp_width, width2), nn.ReLU()]
        block_mlps += [nn.Linear(width2, width4), nn.ReLU()]

        block_mlps += [nn.Linear(width4, 3)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()

#         self.rodriguez = RodriguesModule()

    def forward(self, pose_input):
        out = self.block_mlps(pose_input)
#         Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)
        
        return out
