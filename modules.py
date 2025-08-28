import math

import torch as th
import torch.nn as nn

class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)

def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# def cond_embedding(cond, out_channels, use_scale_shift_norm=False):
#     #print(cond)
#     emb_size = []
#     emb_loc = []
#     cond = cond.cpu()
#     size_layer1= linear(1 ,2 * out_channels if use_scale_shift_norm else out_channels,)
#     size_layer2= linear(2 ,2 * out_channels if use_scale_shift_norm else out_channels,)
#     loc_layer1 = linear(2 ,2 * out_channels if use_scale_shift_norm else out_channels,)
#     loc_layer2 = linear(4 ,2 * out_channels if use_scale_shift_norm else out_channels,)
#     for i in range(cond.shape[0]):
#         if int(cond[i][6])== 1:
#             emb_size.append(size_layer1(cond[i][0].unsqueeze(0)))
#             emb_loc.append(loc_layer1(cond[i][2:4].unsqueeze(0).squeeze()))
#         elif int(cond[i][7]) == 1:
#             emb_size.append(size_layer2(cond[i][0:2].unsqueeze(0).squeeze()))
#             emb_loc.append(loc_layer2(cond[i][2:6].unsqueeze(0).squeeze()))
#         else:            
#             emb_size.append(th.zeros(out_channels))
#             emb_loc.append(th.zeros(out_channels))
#     #print(th.stack(emb_size), th.cat(emb_loc, dim=0)) 
#     return th.stack(emb_size), th.stack(emb_loc)

def checkpoint(func, inputs, params, flag):
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with th.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with th.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = th.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads
    