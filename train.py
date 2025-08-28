import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules import *
from data_preprocessing import *
from MaskControlUNet import *
from Diffusion import *

from params import *
args = parse_arguments()


def main():
    args = parse_arguments()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    images, size_label, location_label, count_label = process_directory(directory_path=args.data_path, size_threshold=args.st)
    if args.norm == 'min_max':
        norm_size_label = min_max_normalize(size_label)
        norm_location_label = min_max_normalize(location_label)
    elif args.norm == 'max':
        norm_size_label = max_normalize(size_label)
        norm_location_label = max_normalize(location_label)
    elif args.norm == 'decimal_scaling':
        norm_size_label = decimal_scaling_normalize(size_label)
        norm_location_label = decimal_scaling_normalize(location_label)
    else:
        norm_size_label = [size/(args.image_size*args.image_size) for size in size_label]
        norm_location_label = [tuple(loc/args.image_size for loc in locs) for locs in location_label]

    print(type(images), type(norm_size_label), type(norm_location_label))
    print(images[0].shape, norm_size_label[0], norm_location_label[0])
    transforming_mask = transforms.Compose([
        transforms.ToTensor(),  
    ])
    normmaskdataset = NormMaskDataset(images, norm_size_label, norm_location_label, transform=transforming_mask)
    batch_size = args.batch_size
    maskdataloader = DataLoader(normmaskdataset, batch_size=batch_size, shuffle=True)

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
    
    model = Diffusion(nn_model=unet, betas=(args.beta1, args.beta2), n_T=args.n_T, device=device, drop_prob=args.dp)
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lrate)

    for ep in range(args.n_epoch+1):
        print(f'Epoch {ep+1}')
        model.train()
        optim.param_groups[0]['lr'] = args.lrate*(1-ep/args.n_epoch)

        pbar = tqdm(maskdataloader)
        loss_ema = None
        for x, conds in pbar:
            optim.zero_grad()
            x = x.to(device)
            conds = conds.to(device)
            loss = model(x, conds)
            
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = args.loss_decay * loss_ema + (1-args.loss_decay) * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            optim.step()

if __name__ == "__main__":
    main()
