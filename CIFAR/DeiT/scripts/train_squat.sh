#!/usr/bin/env bash
set -euo pipefail

die() { echo "Error: $*" 1>&2; exit 2; }

dataset=""
model_size=""
bit=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="${2:-}"; shift 2 ;;
    --model-size) model_size="${2:-}"; shift 2 ;;
    --bit) bit="${2:-}"; shift 2 ;;
    -h|--help) exit 0 ;;
    *) die "Unknown argument: $1" ;;
  esac
done

[[ -n "$dataset" ]] || die "--dataset is required"
[[ -n "$model_size" ]] || die "--model-size is required"
[[ -n "$bit" ]] || die "--bit is required"

case "$dataset" in
  cifar10) dataset_arg="torch/cifar10"; data_dir="./data/CIFAR10"; num_classes="10" ;;
  cifar100) dataset_arg="torch/cifar100"; data_dir="./data/CIFAR100"; num_classes="100" ;;
  *) die "Unsupported --dataset: $dataset" ;;
esac

case "$model_size" in
  tiny) model_name="deit_tiny_distilled_patch16_224" ;;
  small) model_name="deit_small_distilled_patch16_224" ;;
  *) die "Unsupported --model-size: $model_size" ;;
esac

[[ "$bit" =~ ^[0-9]+$ ]] || die "--bit must be an integer"
wbits="$bit"
abits="$bit"
feature_levels="$bit"

config_file="configs/${dataset}_deit_${model_size}_squat.yml"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "${script_dir}/.." && pwd)"
cd "$repo_dir"

python train_squat.py \
  --config "${config_file}" \
  "${data_dir}" \
  --dataset "${dataset_arg}" \
  --model "${model_name}" \
  --num-classes "${num_classes}" \
  --img-size 224 \
  --batch-size 128 \
  --epochs 300 \
  --opt adamw \
  --lr 0.001 \
  --weight-decay 0.05 \
  --sched cosine \
  --warmup-lr 0.0001 \
  --min-lr 0.00001 \
  --warmup-epochs 10 \
  --wq-enable \
  --wq-mode statsq \
  --wq-bitw "${wbits}" \
  --aq-enable \
  --aq-mode lsq \
  --aq-bitw "${abits}" \
  --qmodules blocks \
  --use-squat \
  --QFeatureFlag \
  --feature-levels "${feature_levels}" \
  --use-student-quant-params \
  --kd-T 4.0 \
  --kd-beta 1.0 \
  --kd-gamma 1.0 \
  --feature-distill-loss-type L2 \
  --distill-token-mode all_tokens \
  --seed 42 \
  --gpu-id 0 \
  --workers 4

