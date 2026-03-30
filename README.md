# SQuaT

This repository is the official implementation of **"SQuaT: Self-Supervised Knowledge Distillation via Student-Aware Quantized Teacher Features"** (AISTATS 2026) by [HyeonJun Lee*](https://github.com/lcdbsa522), [Hyeonsik Jo*](https://github.com/hsjo827), Jinwoo Chung, and [Jangho Kim†](https://github.com/Jangho-Kim)

![figure_2](https://github.com/user-attachments/assets/24ab6be2-61a5-4dfb-afba-8408f3d4523b)

## Installation

```bash
conda create -n squat python=3.10 -y
conda activate squat
pip install -r requirements.txt
```

## Training

### CIFAR (ResNet-20, 32 / DeiT)

```bash
# ResNet
cd CIFAR/ResNet
bash ./scripts/run_cifar10_resnet20_squat.sh "fp/"
bash ./scripts/run_cifar10_resnet20_squat.sh "SQuaT/W1A1/"

# DeiT (Vision Transformer)
cd CIFAR/DeiT
bash ./scripts/train_teacher.sh --dataset cifar100 --model-size tiny
bash ./scripts/train_squat.sh --dataset cifar100 --model-size tiny --bit 4
```

### ImageNet (ResNet-18)

```bash
cd ImageNet
bash ./scripts/run_SQuaT.sh "SQuaT/W1A1/"
```

### GLUE (BERT)
Download the datasets and FP weights from [GLUE](https://gluebenchmark.com/) and [HuggingFace](https://huggingface.co/JeremiahZ), then place them in the `data/` and `models/` folders within the `GLUE` directory.

```bash
cd GLUE
# Usage 
bash train_squat.sh <task_name> <bert_size> <bit> <gpu_ids> [num_gpus]
# Example
bash train_squat.sh "rte" "base" “3” "0" 1
```

## Citation
Please refer to the following citation if this repository is useful for your research.

```bibtex
@inproceedings{squat2025,
  title={},
  author={},
  booktitle={},
  year={}
}
```
