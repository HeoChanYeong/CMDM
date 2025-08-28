# CMDM: Controllable Mask Diffusion Model for Medical Annotation Synthesis with Semantic Information Extraction

[[Paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010482525011588)] 

This is the official implementation of the paper “Controllable Mask Diffusion Model for Medical Annotation Synthesis with Semantic Information Extraction”, published in **Computers in Biology and Medicine**.

<p align="center">
<img src=/assets/cibm_git.png />
</p>

## Table of Contents
- [Requirements](#requirements)
- [Dataset Preparation](#dataset-preparation)
- [Training Your Own CMDM](#training-your-own-cmdm)
- [Sampling with CMDM](#sampling-with-cmdm)
- [Acknowledgement](#acknowledgement)
- [Citations](#citations)

## Requirements
```bash
conda create -n CMDM python=3.8.10
conda activate CMDM
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

## Dataset Preparation
The proposed framework requires medical annotatoin data.

Please organize the dataset with the following structure:
```angular2
├── ${data_root}
│ ├── ${train_dataset_dir}
│ │ ├── masks
│ │ │ ├── ***.png
```

## Training Your Own CMDM
To train your own CMDM, Run the following command:

```bash
python train.py --data_path ./TrainDataset \
               --image_size 256 \
               --n_epoch 2000 \
               --n_T 500 \
               --batch_size 2 \
```

## Sampling with CMDM
To sample with CMDM, set the number of samples you need via ```n_samples``` in ```sampling_si.py```, then run:
```bash
python sampling_mask.py
```
As proposed in the paper, ```sampling_si.py``` first analyzes the correlations among semantic information (e.g., size and location) within the given annotation masks, and ```sampling_mask.py``` then generates the corresponding annotation masks.

## Acknowledgement
This repository is based on [LDM](https://github.com/CompVis/latent-diffusion), [guided-diffusion](https://github.com/openai/guided-diffusion), [CFG](https://github.com/TeaPearce/Conditional_Diffusion_MNIST) and [SDM](https://github.com/WeilunWang/semantic-diffusion-model). We sincerely thank the original authors for their valuable contributions and outstanding work.


## Citations
```
@article{heo2025controllable,
  title={Controllable Mask Diffusion Model for medical annotation synthesis with semantic information extraction},
  author={Heo, Chanyeong and Jung, Jaehee},
  journal={Computers in Biology and Medicine},
  volume={196},
  pages={110807},
  year={2025},
  publisher={Elsevier}
  doi={https://doi.org/10.1016/j.compbiomed.2025.110807}}
```
