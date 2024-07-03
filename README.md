[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on-20)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-20?p=dhr-dual-features-driven-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=dhr-dual-features-driven-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on-21)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-21?p=dhr-dual-features-driven-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on-22)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-22?p=dhr-dual-features-driven-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=dhr-dual-features-driven-hierarchical)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dhr-dual-features-driven-hierarchical/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=dhr-dual-features-driven-hierarchical)

# DHR: Dual Features-Driven Hierarchical Rebalancing in Inter- and Intra-Class Regions for Weakly-Supervised Semantic Segmentation
This repository is the official implementation of "DHR: Dual Features-Driven Hierarchical Rebalancing in Inter- and Intra-Class Regions for Weakly-Supervised Semantic Segmentation".

[arXiv](https://arxiv.org/abs/2404.00380)

# Update
[07/02/2024] Our DHR has been accepted to ECCV 2024. 🔥🔥🔥

[04/02/2024] Released initial commits.

### Citation
Please cite our paper if the code is helpful to your research.
```
@inproceedings{jo2024dhr,
      title={DHR: Dual Features-Driven Hierarchical Rebalancing in Inter- and Intra-Class Regions for Weakly-Supervised Semantic Segmentation}, 
      author={Sanghyun Jo and Fei Pan and In-Jae Yu and Kyungsu Kim},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2024}
}
```

### Abstract
Weakly-supervised semantic segmentation (WSS) ensures high-quality segmentation with limited data and excels when employed as input seed masks for large-scale vision models such as Segment Anything. However, WSS faces challenges related to minor classes since those are overlooked in images with adjacent multiple classes, a limitation originating from the overfitting of traditional expansion methods like Random Walk. We first address this by employing unsupervised and weakly-supervised feature maps instead of conventional methodologies, allowing for hierarchical mask enhancement. This method distinctly categorizes higher-level classes and subsequently separates their associated lower-level classes, ensuring all classes are correctly restored in the mask without losing minor ones. Our approach, validated through extensive experimentation, significantly improves WSS across five benchmarks (VOC: 79.8\%, COCO: 53.9\%, Context: 49.0\%, ADE: 32.9\%, Stuff: 37.4\%), reducing the gap with fully supervised methods by over 84\% on the VOC validation set.

![Overview](./figures/Overview.jpg)

# Setup

Setting up for this project involves installing dependencies and preparing datasets. The code is tested on Ubuntu 20.04 with NVIDIA GPUs and CUDA installed. 

### Installing dependencies
To install all dependencies, please run the following:
```bash
pip install -U "ray[default]"
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
python3 -m pip install -r requirements.txt
```

or reproduce our results using docker.
```bash
docker build -t dhr_pytorch:v1.13.1 .
docker run --gpus all -it --rm \
--shm-size 32G --volume="$(pwd):$(pwd)" --workdir="$(pwd)" \
dhr_pytorch:v1.13.1
```

### Preparing datasets

Please download following VOC, COCO, Context, ADE, and COCO-Stuff datasets. Each dataset has a different directory structure. Therefore, we modify directory structures of all datasets for a comfortable implementation. 

> ##### 1. PASCAL VOC 2012
> Download PASCAL VOC 2012 dataset from our [[Google Drive](https://drive.google.com/file/d/1ITnF19LayDdC1QYUki1guta82L9qSrGo/view?usp=sharing)].

> ##### 2. MS COCO 2014
> Download MS COCO 2014 dataset from our [[Google Drive](https://drive.google.com/file/d/1WwcK-33wHpGw4cozEi7hkmpI0GuYoPEc/view?usp=sharing)].

> ##### 3. Pascal Context
> Download Pascal Context dataset from our [[Google Drive](https://drive.google.com/file/d/1OSMRUjSl-o7u_BMgp83_0PEJMLKue5VJ/view?usp=sharing)].

> ##### 4. ADE 2016
> Download ADE 2016 dataset from our [[Google Drive](https://drive.google.com/file/d/11I9bD1X_6KXh-I3oQTIiUZrPVBlQ56OB/view?usp=sharing)].

> ##### 5. COCO-Stuff
> Download COCO-Stuff dataset from our [[Google Drive](https://drive.google.com/file/d/1tFy3RWy9DsME8cNM8jC-rlsleHF9kySc/view?usp=sharing)].

> ##### 6. Open-vocabulary Segmentation Models
> Download [[all results](https://drive.google.com/file/d/1-8KCNa2qhE0vjsXsqzsB8eb5dZ1oiFA_/view?usp=sharing)] and [[the reproduced project](https://drive.google.com/file/d/155e9GEPJN3Uub88IEjj-Qeto7Eqolamc/view?usp=sharing)] for a fair comparison with WSS.

Create a directory "../VOC2012/" for storing the dataset and appropriately place each dataset to have the following directory structure.
```
    ../                               # parent directory
    ├── ./                            # current (project) directory
    │   ├── core/                     # (dir.) implementation of our DHR (e.g., OT)
    │   ├── tools/                    # (dir.) helper functions
    │   ├── experiments/              # (dir.) checkpoints and WSS masks
    │   ├── README.md                 # instruction for a reproduction
    │   └── ... some python files ...
    │
    ├── WSS/                          # WSS masks across all training and testing datasets
    │   ├── VOC2012/          
    │   │   ├── RSEPM/        
    │   │   ├── MARS/
    │   │   └── DHR/
    │   ├── COCO2014/
    │   │   └── DHR/
    │   ├── PascalContext/
    │   │   └── DHR/
    │   ├── ADE2016/   
    │   │   └── DHR/
    │   └── COCO-Stuff/
    │       └── DHR/
    │
    ├── GroundingDINO_Ferret_SAM/     # reproduced project for Grounding DINO and Ferret with SAM
    │   ├── core/                     # (dir.) implementation details
    │   ├── tools/                    # (dir.) helper functions
    │   ├── weights/                  # (dir.) checkpoints of Grounding DINO and Ferret
    │   ├── README.md                 # instruction for implementing Grounding DINO and Ferret
    │   └── ... some python files ...
    │
    ├── OVSeg/                        # SAM-based outputs of Grounding DINO and Ferret for a fair comparison
    │   ├── VOC2012/      
    │   │   ├── GroundingDINO+SAM/
    │   │   └── Ferret+SAM/
    │   ├── COCO2014/
    │   │   ├── GroundingDINO+SAM/
    │   │   └── Ferret+SAM/
    │   ├── PascalContext/
    │   │   ├── GroundingDINO+SAM/
    │   │   └── Ferret+SAM/
    │   ├── ADE2016/   
    │   │   ├── GroundingDINO+SAM/
    │   │   └── Ferret+SAM/
    │   └── COCO-Stuff/
    │       ├── GroundingDINO+SAM/
    │       └── Ferret+SAM/
    │
    ├── VOC2012/                      # PASCAL VOC 2012
    │   ├── train_aug/
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/   
    │   ├── validation/
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/   
    │   └── test/
    │       └── image/
    │
    ├── COCO2014/                     # MS COCO 2014
    │   ├── train/              
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/
    │   └── validation/
    │       ├── image/     
    │       ├── mask/        
    │       └── xml/
    │
    ├── PascalContext/                # PascalContext
    │   ├── train/              
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/
    │   └── validation/
    │       ├── image/     
    │       ├── mask/        
    │       └── xml/
    │
    ├── ADE2016/                      # ADE2016
    │   ├── train/              
    │   │   ├── image/     
    │   │   ├── mask/        
    │   │   └── xml/
    │   └── validation/
    │       ├── image/     
    │       ├── mask/        
    │       └── xml/
    │
    └── COCO-Stuff/                   # COCO-Stuff
        ├── train/              
        │   ├── image/     
        │   ├── mask/        
        │   └── xml/
        └── validation/
            ├── image/     
            ├── mask/        
            └── xml/
```

# Preprocessing
### 1. Training the USS method
Please download the trained CAUSE weights from scratch on other datasets [CAUSE weights](https://drive.google.com/file/d/1A8qDMeiF6i_8gNI6At5R5NSS21i93rUi/view?usp=sharing).
We follow the official [CAUSE](https://github.com/byungkwanlee/causal-unsupervised-segmentation) to train CAUSE from scratch on five datasets.

### 2. Training the WSS method
Please download and prepare WSS masks [WSS labels](https://drive.google.com/file/d/1fKX2OFvVcpgqmiibZh-t56PeEc_zIa22/view?usp=sharing).
You can replace existing WSS methods with other WSS methods following the current structure.

# Training
Our code is coming soon.

<!-- Extract USS feature maps using the frozen USS checkpoint.
```bash
python3 produce_uss_features.py --gpus 0 --root ../ --data VOC2012 --domain train_aug --uss CAUSE
```

Train our DHR with WSS labels and USS features.
```bash
python3 train.py --gpus 0 --root ../ --dataset VOC2012 --train_domain train_aug --valid_domain validation \
--backbone resnet101 --decoder deeplabv3+ --wss MARS --uss CAUSE --tau 0.8 \
--tag "ResNet-101@VOC2012@DeepLabv3+@DHR"
``` -->

# Evaluation
Release our checkpoint and official VOC results (anonymous links).

| Method | Backbone     | Checkpoints                  | VOC val | VOC test |
|:------:|:------------:|:----------------------------:|:-------:|:--------:|
| DHR    | ResNet-101   | [Google Drive](https://drive.google.com/file/d/1i-1x1VFYUqHqB_uUAg7C-HmvrLxx3l-h/view?usp=sharing) | [link](http://host.robots.ox.ac.uk:8080/anonymous/A4RUI9.html) | [link](http://host.robots.ox.ac.uk:8080/anonymous/HICQUU.html) |

Below lines are testing commands to reproduce our results.
Additionally, we follow the official [Mask2Former](https://github.com/facebookresearch/Mask2Former) to train Swin-L+Mask2Former with our DHR masks on five datasets.
```bash
# Generate the final segmentation outputs with CRF
python3 produce_wss_masks.py --gpus 0 --cpus 64 --root ../ --data VOC2012 --domain validation \
--backbone resnet101 --decoder deeplabv3+ --tag "ResNet-101@VOC2012@DeepLabv3+@DHR" --checkpoint "last"

# Calculate the mIoU
python3 evaluate.py --fix --data VOC2012 --gt ../VOC2012/validation/mask/ \
--tag "DHR" --pred "./experiments/results/VOC2012/ResNet-101@VOC2012@DeepLabv3+@DHR@last/validation/"

# Reproduce WSS performance related to official VOC results
#            DHR (Ours, DeepLabv3+) | mIoU: 79.6%, mFPR: 0.127, mFNR: 0.077
#           DHR (Ours, Mask2Former) | mIoU: 81.7%, mFPR: 0.131, mFNR: 0.052
python3 evaluate.py --fix --data VOC2012 --gt ../VOC2012/validation/mask/ \
--tag "DHR (Ours, DeepLabv3+)" --pred "./submissions_DHR@DeepLabv3+/validation/results/VOC2012/Segmentation/comp5_val_cls/"
python3 evaluate.py --fix --data VOC2012 --gt ../VOC2012/validation/mask/ \
--tag "DHR (Ours, Mask2Former)" --pred "./submissions_DHR@Mask2Former/validation/results/VOC2012/Segmentation/comp5_val_cls/"
```