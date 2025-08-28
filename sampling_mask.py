from abc import abstractmethod
import math
import torch
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torchvision import datasets, transforms, utils, models
from torchvision.utils import save_image, make_grid
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Subset
from torch.utils import data

#import import_ipynb
from modules import *
from MaskControlUNet import *
from Diffusion import *
from train import *

from params import *
args = parse_arguments()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

unet = create_model(
        args.image_size, 
        args.num_classes, 
        args.num_channels, 
        args.num_res_blocks, 
        channel_mult=args.channel_mult, 
        learn_sigma=args.learn_sigma, 
        class_cond=args.class_cond, 
        use_checkpoint=args.use_checkpoint, 
        attention_resolutions=args.attention_resolutions,
        num_heads=args.num_heads,
        num_head_channels=args.num_head_channels,
        num_heads_upsample=args.num_heads_upsample,
        use_scale_shift_norm=args.use_scale_shift_norm,
        dropout=args.dropout,
        resblock_updown=args.resblock_updown,
        use_fp16=args.use_fp16,
        use_new_attention_order=args.use_new_attention_order,
        no_instance=args.no_instance,
    )

diffusion = Diffusion(nn_model=unet, betas=(args.beta1, args.beta2), n_T=args.n_T, device=device, drop_prob=args.dp)
optim = torch.optim.Adam(diffusion.parameters(), lr=args.lrate)

save_dir = args.save_dir
checkpoint = torch.load(save_dir)  
diffusion.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

def convert_data(traditional_coords):
    print(traditional_coords)
    converted_coords = []
    for x, y in traditional_coords:
        converted_y = args.image_size - y  
        converted_coords.append((x, converted_y))
    return converted_coords


def mask_saving(test_cond,count,w):
    gen_list = []
    diffusion.eval()
    with torch.no_grad():
        n_sample = 1
        for i in range(len(test_cond)):
            label = torch.tensor(test_cond[i])
        
            x_gen = diffusion.sample(n_sample, (1,args.image_size,args.image_size), device, condition=label.unsqueeze(0), guide_w=w) #ws_test의 guidance값과 class의 마다 n_sample개씩 생성
            x_gen = x_gen.clamp(0,1)
            gen_list.append(x_gen)
    #Repeated Reverse Process
    temp = torch.zeros((1, 256, 256)).to(device)
    for i in range(count):
        temp += gen_list[i].squeeze(0)

    mask = temp.clamp(0.1)
    mask = mask.cpu().numpy().transpose(1,2,0).squeeze()