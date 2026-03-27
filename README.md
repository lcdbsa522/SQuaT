# SQuaT: Student-Quantized Teacher for Feature Distillation

Official PyTorch implementation of **SQuaT** — a feature distillation framework that bridges the representation gap between full-precision teachers and quantized students by quantizing teacher features using the student's own quantization parameters.

<!-- If you have a paper link, uncomment and fill in:
> **[Paper Title](https://arxiv.org/abs/xxxx.xxxxx)**  
> Author1, Author2, ...  
> *Conference/Journal, Year*
-->

## Overview

Low-bit quantization significantly reduces model size and inference cost, but the accuracy gap between full-precision and quantized models remains a challenge. Traditional knowledge distillation methods suffer from a **representation gap** — the teacher produces full-precision features while the student operates in a quantized space, making direct feature matching suboptimal.

**SQuaT** addresses this by introducing a *Student-Quantized Teacher*: the teacher's intermediate features are quantized using the student's learned quantization parameters before distillation. This ensures both teacher and student features reside in a compatible representation space, leading to more effective knowledge transfer.

### Key Features

- **Feature-level distillation** with quantization-aware teacher representations
- **Hessian-based gradient scaling** (EWGS) for improved quantization-aware training
- Support for **ultra-low bit-widths** (1-bit, 2-bit weights and activations)
- Experiments across **4 domains**: CIFAR, ImageNet, DeiT (Vision Transformer), and BERT (NLP)

## Project Structure

```
SQuaT/
├── CIFAR/          # ResNet / VGG on CIFAR-10, CIFAR-100
├── ImageNet/       # ResNet-18 on ILSVRC2012
├── DeiT/           # DeiT (Vision Transformer) quantization
├── BERT/           # BERT quantization on GLUE tasks
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/SQuaT.git
cd SQuaT
pip install -r requirements.txt
```

> **Note**: PyTorch with CUDA support should be installed separately based on your environment.  
> See [PyTorch installation guide](https://pytorch.org/get-started/locally/).

## Quick Start

### CIFAR (ResNet-20, W1A1)

```bash
cd CIFAR

# 1. Train full-precision model
python train_fp.py --dataset cifar10 --arch resnet20_fp

# 2. Train quantized student with SQuaT distillation
python train_quant_distill.py \
    --dataset cifar10 \
    --arch resnet20_quant \
    --teacher_arch resnet20_fp \
    --distill fd \
    --weight_levels 2 --act_levels 2 \
    --QFeatureFlag True \
    --load_pretrain True \
    --pretrain_path ./results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth \
    --teacher_path ./results/CIFAR10_ResNet20/fp/checkpoint/last_checkpoint.pth
```

### ImageNet (ResNet-18)

```bash
cd ImageNet

python ImageNet_train_quant.py \
    --data /path/to/ILSVRC2012 \
    --arch resnet18_quant \
    --teacher_arch resnet18_fp \
    --distill_type fd \
    --weight_levels 2 --act_levels 2 \
    --QFeatureFlag True
```

### DeiT (Vision Transformer)

```bash
cd DeiT

python train_squat.py \
    --data_dir /path/to/imagenet \
    --model deit_tiny_patch16_224 \
    --teacher deit_tiny_patch16_224 \
    --teacher-pretrained \
    --use-squat \
    --wq-enable --wq-bitw 2 \
    --aq-enable --aq-bitw 2 \
    --QFeatureFlag
```

### BERT (GLUE Tasks)

```bash
cd BERT

python main.py \
    --task_name sst-2 \
    --weight_bits 2 --input_bits 8 \
    --squat_distill True
```

## Method

<p align="center"><i>SQuaT quantizes teacher features using the student's quantization parameters, aligning both into a shared representation space for effective feature distillation.</i></p>

1. **Student model**: Weights and activations are quantized to low bit-widths via EWGS
2. **Teacher model**: Full-precision model whose intermediate features are extracted
3. **Feature quantization**: Teacher features are quantized using the student's learned quantization parameters (step sizes)
4. **Feature distillation**: The quantized teacher features are matched with the student's quantized features via L1/L2/KL loss

## Requirements

- Python ≥ 3.6
- PyTorch (see `requirements.txt` for full details)
- CUDA-capable GPU

## Citation

<!--
If you find this work useful, please cite:

```bibtex
@inproceedings{squat2025,
  title={},
  author={},
  booktitle={},
  year={}
}
```
-->

## Acknowledgements

This codebase builds upon [EWGS](https://github.com/cvlab-yonsei/EWGS) for quantization-aware training.

## License

<!-- Specify your license here -->