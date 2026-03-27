#!/bin/bash


#########################################################################################################
# Dataset: CIFAR-100
# Model: ResNet-32
# 'weight_levels' and 'act_levels' and 'feature_levels' correspond to 2^b, where b is a target bit-width.

# Method: SQuaT+EWGS
# Bit-width: W1A1, W2A2, W4A4

# Usage: ./run_cifar100_resnet32_squat.sh METHOD_TYPE
# Example: ./run_cifar100_resnet32_squat.sh W1A1
#########################################################################################################

set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING RUN AT $start_fmt"

METHOD_TYPE=$1
echo "Method Type: $METHOD_TYPE"

# ===========================================
# cifar100
# resnet32
# ===========================================

# fp
if [ $METHOD_TYPE == "fp/" ] 
then
    python train_fp.py --gpu_id '0' \
                    --dataset 'cifar100' \
                    --arch 'resnet32_fp' \
                    --num_workers 8 \
                    --batch_size 64 \
                    --lr_m 0.05 \
                    --weight_decay 5e-4 \
                    --lr_scheduler_m 'cosine' \
                    --epochs 720 \
                    --log_dir './results/CIFAR100_ResNet32/'$METHOD_TYPE

# W1A1 (Weight: 2 levels, Act: 2 levels, Feature: 2 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W1A1/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar100' \
                                --arch 'resnet32_quant' \
                                --teacher_arch 'resnet32_fp' \
                                --epochs 720 \
                                --batch-size 64 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 5e-4 \
                                --lr_q 5e-6 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_decay 5e-4 \
                                --weight_levels 2 \
                                --act_levels 2 \
                                --feature_levels 2 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 10 \
                                --QFeatureFlag True \
                                --use_student_quant_params True \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill 'fd' \
                                --distill_loss 'L2' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --pretrain_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir "./results/CIFAR100_ResNet32/${METHOD_TYPE}" \
                                --resume "./results/CIFAR100_ResNet32/${METHOD_TYPE}/checkpoint/last_checkpoint.pth"

# W2A2 (Weight: 4 levels, Act: 4 levels, Feature: 4 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W2A2/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar100' \
                                --arch 'resnet32_quant' \
                                --teacher_arch 'resnet32_fp' \
                                --epochs 720 \
                                --batch-size 64 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 5e-4 \
                                --lr_q 5e-6 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_decay 5e-4 \
                                --weight_levels 4 \
                                --act_levels 4 \
                                --feature_levels 4 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 10 \
                                --QFeatureFlag True \
                                --use_student_quant_params True \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill 'fd' \
                                --distill_loss 'L2' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --pretrain_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir "./results/CIFAR100_ResNet32/${METHOD_TYPE}" \
                                --resume "./results/CIFAR100_ResNet32/${METHOD_TYPE}/checkpoint/last_checkpoint.pth"

# W4A4 (Weight: 16 levels, Act: 16 levels, Feature: 16 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W4A4/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar100' \
                                --arch 'resnet32_quant' \
                                --teacher_arch 'resnet32_fp' \
                                --epochs 720 \
                                --batch-size 64 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 5e-4 \
                                --lr_q 5e-6 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_decay 5e-4 \
                                --weight_levels 16 \
                                --act_levels 16 \
                                --feature_levels 16 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 10 \
                                --QFeatureFlag True \
                                --use_student_quant_params True \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill 'fd' \
                                --distill_loss 'L2' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --pretrain_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR100_ResNet32/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir "./results/CIFAR100_ResNet32/${METHOD_TYPE}" \
                                --resume "./results/CIFAR100_ResNet32/${METHOD_TYPE}/checkpoint/last_checkpoint.pth"
                                

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

total_time=$(( $end - $start ))
echo "RESULT: Total run time: $total_time seconds, started at $start_fmt"
