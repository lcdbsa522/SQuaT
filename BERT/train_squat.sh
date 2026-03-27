#!/bin/bash

# Quantization Option
act_quant=1
weight_quant=1
bits=${3:-""}
if [ -z "$bits" ]; then
    echo "Error: bit (3rd arg) is required. Example: bash train_squat.sh \"rte\" \"base\" \"2\" \"0\" 1"
    exit 2
fi
case "$bits" in
    1|2|3|4|8) ;;
    *) echo "Error: invalid bit: $bits"; exit 2 ;;
esac
weight_bits=$bits
input_bits=$bits

# KD options
gt_loss=0
pred_distill=1
rep_distill=0
attn_distill=0
attnmap_distill=0
context_distill=0
output_distill=0
sa_output_distill=0

# SQuaT options
squat_distill=1
squat_coeff=1.0
use_adaptor=0
use_adaptor_bn=0
squat_layer=-1  # -1: last layer, -2: all layers
squat_token=all  # cls or all

# BERT option (base, large)
bert=$2

# GPU settings
gpu_ids=$4
num_gpus=${5:-1}

# DA options
aug_train=0
aug_N=30

# Training Options
learning_rate=2E-5
num_train_epochs=3
task_name=$1
neptune=0
save_quantized_model=1

# ============================================= #
exp_name=${task_name}_${bert}_squat_w${bits}a${bits}

echo "=========================================="
echo "Training with ${bits}-bit quantization (W${bits}A${bits})"
echo "SQuaT: Enabled"
echo "Experiment: ${exp_name}"
echo "=========================================="

CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --data_dir data --task_name $task_name --output_dir output --num_train_epochs $num_train_epochs --bert ${bert} \
--weight_bits ${weight_bits} --input_bits ${input_bits} --gpus ${num_gpus} --act_quant ${act_quant} --weight_quant ${weight_quant} --aug_train ${aug_train} --aug_N ${aug_N} \
--sa_output_distill $sa_output_distill --output_distill ${output_distill} --context_distill ${context_distill} --gt_loss ${gt_loss} --pred_distill ${pred_distill} --rep_distill ${rep_distill} --attn_distill ${attn_distill} --attnmap_distill ${attnmap_distill} \
--squat_distill ${squat_distill} --squat_coeff ${squat_coeff} --use_adaptor ${use_adaptor} --use_adaptor_bn ${use_adaptor_bn} --squat_layer ${squat_layer} --squat_token ${squat_token} \
--exp_name ${exp_name} --save_quantized_model ${save_quantized_model} \
--neptune ${neptune} --learning_rate ${learning_rate}

echo "Completed ${bits}-bit training"

