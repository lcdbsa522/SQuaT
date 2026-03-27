#!/bin/bash


#########################################################################################################
# Dataset: CIFAR-10
# Model: ResNet-20
# 'weight_levels' and 'act_levels' and 'feature_levels' correspond to 2^b, where b is a target bit-width.

# Method: SQuaT+EWGS
# Bit-width: W1A1, W2A2, W4A4

# Usage: ./run_cifar10_resnet20_squat.sh METHOD_TYPE
# Example: ./run_cifar10_resnet20_squat.sh "SQuaT/W1A1/"
#########################################################################################################

set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING RUN AT $start_fmt"

METHOD_TYPE=$1
echo "Method Type: $METHOD_TYPE"

# ===========================================
# cifar10
# resnet20
# ===========================================

# fp
if [ $METHOD_TYPE == "fp/" ] 
then
    python3 train_fp.py --gpu_id '0' \
                    --arch 'resnet20_fp' \
                    --log_dir './results/CIFAR10_ResNet20/'$METHOD_TYPE \
                    --epochs 1200

# W1A1 (Weight: 2 levels, Act: 2 levels, Feature: 2 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W1A1/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar10' \
                                --arch 'resnet20_quant' \
                                --teacher_arch 'resnet20_fp' \
                                --epochs 1200 \
                                --batch-size 256 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-3 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
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
                                --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir './results/CIFAR10_ResNet20/SQuaT/W1A1/' \
                                --resume './results/CIFAR10_ResNet20/SQuaT/W1A1/checkpoint/last_checkpoint.pth'

# W2A2 (Weight: 4 levels, Act: 4 levels, Feature: 4 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W2A2/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar10' \
                                --arch 'resnet20_quant' \
                                --teacher_arch 'resnet20_fp' \
                                --epochs 1200 \
                                --batch-size 256 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-3 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
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
                                --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir './results/CIFAR10_ResNet20/SQuaT/W2A2/' \
                                --resume './results/CIFAR10_ResNet20/SQuaT/W2A2/checkpoint/last_checkpoint.pth'

# W4A4 (Weight: 16 levels, Act: 16 levels, Feature: 16 levels)
elif [ "$METHOD_TYPE" == "SQuaT/W4A4/" ]
then
    python train_quant_distill.py --gpu_id '0' \
                                --workers 4 \
                                --dataset 'cifar10' \
                                --arch 'resnet20_quant' \
                                --teacher_arch 'resnet20_fp' \
                                --epochs 1200 \
                                --batch-size 256 \
                                --optimizer_m 'Adam' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-3 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
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
                                --pretrain_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --teacher_path './results/CIFAR10_ResNet20/fp/checkpoint/best_checkpoint.pth' \
                                --log_dir './results/CIFAR10_ResNet20/SQuaT/W4A4/' \
                                --resume './results/CIFAR10_ResNet20/SQuaT/W4A4/checkpoint/last_checkpoint.pth'
                                

end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"

total_time=$(( $end - $start ))
echo "RESULT: Total run time: $total_time seconds, started at $start_fmt"
