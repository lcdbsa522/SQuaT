#!/bin/bash


#########################################################################################################
# Dataset: ImageNet-1K
# Model: ResNet-18
# 'weight_levels' and 'act_levels' and 'feature_levels' correspond to 2^b, where b is a target bit-width.

# Method: SQuaT+EWGS
# Bit-width: W1A1, W3A3, W4A4, W8A8

# Usage: ./run_SQuaT.sh BIT_LEVEL
# Example: ./run_SQuaT.sh "SQuaT/W1A1/"
#########################################################################################################

set -e

# Start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING RUN AT $start_fmt"

BIT_LEVEL=$1

echo "Bit Levels: $BIT_LEVEL"

# W1A1 (Weight: 2 levels, Act: 2 levels, Feature: 2 levels)
if [ "$BIT_LEVEL" == "SQuaT/W1A1/" ]
then
    python ImageNet_train_quant.py --gpu 0 \
                                --visible_gpus '0' \
                                --workers 8 \
                                --data '/path/to/ILSVRC2012' \
                                --arch 'resnet18_quant' \
                                --teacher_arch 'resnet18_fp' \
                                --epochs 100 \
                                --batch-size 128 \
                                --optimizer_m 'SGD' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-2 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_levels 2 \
                                --act_levels 2 \
                                --feature_levels 2 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 1 \
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
                                --log_dir './results/ResNet18/SQuaT/W1A1' \
                                --resume './results/ResNet18/SQuaT/W1A1/checkpoint.pth.tar'

# W3A3 (Weight: 8 levels, Act: 8 levels, Feature: 8 levels)
elif [ "$BIT_LEVEL" == "SQuaT/W3A3/" ]
then
    python ImageNet_train_quant.py --gpu 0 \
                                --visible_gpus '0' \
                                --workers 8 \
                                --data '/path/to/ILSVRC2012' \
                                --arch 'resnet18_quant' \
                                --teacher_arch 'resnet18_fp' \
                                --epochs 100 \
                                --batch-size 128 \
                                --optimizer_m 'SGD' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-2 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_levels 8 \
                                --act_levels 8 \
                                --feature_levels 8 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 1 \
                                --QFeatureFlag True \
                                --use_student_quant_params True \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill_type 'fd' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --log_dir './results/ResNet18/SQuaT/W3A3' \
                                --resume './results/ResNet18/SQuaT/W3A3/checkpoint.pth.tar'

# W4A4 (Weight: 16 levels, Act: 16 levels, Feature: 16 levels)
elif [ "$BIT_LEVEL" == "SQuaT/W4A4/" ]
then
    python ImageNet_train_quant.py --gpu 0 \
                                --visible_gpus '0' \
                                --workers 8 \
                                --data '/path/to/ILSVRC2012' \
                                --arch 'resnet18_quant' \
                                --teacher_arch 'resnet18_fp' \
                                --epochs 100 \
                                --batch-size 128 \
                                --optimizer_m 'SGD' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-2 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_levels 16 \
                                --act_levels 16 \
                                --feature_levels 16 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 1 \
                                --QFeatureFlag False \
                                --use_student_quant_params False \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill_type 'fd' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --log_dir './results/ResNet18/SQuaT/W4A4' \
                                --resume './results/ResNet18/SQuaT/W4A4/checkpoint.pth.tar'

# W8A8 (Weight: 256 levels, Act: 256 levels, Feature: 256 levels)
elif [ "$BIT_LEVEL" == "SQuaT/W8A8/" ]
then
    python ImageNet_train_quant.py --gpu 0 \
                                --visible_gpus '0' \
                                --workers 8 \
                                --data '/path/to/ILSVRC2012' \
                                --arch 'resnet18_quant' \
                                --teacher_arch 'resnet18_fp' \
                                --epochs 100 \
                                --batch-size 128 \
                                --optimizer_m 'SGD' \
                                --optimizer_q 'Adam' \
                                --lr_m 1e-2 \
                                --lr_q 1e-5 \
                                --lr_scheduler_m 'cosine' \
                                --lr_scheduler_q 'cosine' \
                                --weight_levels 256 \
                                --act_levels 256 \
                                --feature_levels 256 \
                                --baseline False \
                                --use_hessian True \
                                --update_scales_every 1 \
                                --QFeatureFlag False \
                                --use_student_quant_params False \
                                --use_adapter False \
                                --use_adapter_bn False \
                                --model_type 'student' \
                                --distill_type 'fd' \
                                --kd_T 4 \
                                --kd_alpha 0.0 \
                                --kd_beta 1.0 \
                                --kd_gamma 1.0 \
                                --log_dir './results/ResNet18/SQuaT/W8A8' \
                                --resume './results/ResNet18/SQuaT/W8A8/checkpoint.pth.tar'

else
    echo "Unknown BIT_LEVEL: $BIT_LEVEL"
    exit 1
fi

# End timing
end=$(date +%s)
echo "ENDING RUN AT $(date +%Y-%m-%d\ %r)"
echo "Total time: $(( end - start )) seconds"