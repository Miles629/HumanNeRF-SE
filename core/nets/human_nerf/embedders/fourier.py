import torch
import torch.nn as nn
#傅里叶？
class Embedder:
    def __init__(self, **kwargs):#keyword argument **表示创建字典
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):#self为类的实操对象
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)#lambda：return x,append 在列表末尾再增加一个
            out_dim += d#输入include_input 为true时，out_dim=input_dims
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        #linspace start=0. end=max_freq,steps=分割的点数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)
    #把多个tensor进行拼接


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'periodic_fns' : [torch.sin, torch.cos],#激活函数
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
