#!/bin/bash

root="../"
dataset_dir=/public2/zhongyutian/Liuzhenyang/VasNet-master/input/IVG/
result_dir=/public2/zhongyutian/Liuzhenyang/VasNet-master/output_test/IVG
model_dir=/public2/zhongyutian/Liuzhenyang/VasNet-master/output_train/Fangzhen-400_352_crop_and_rotate_and_scale/00299.pth

resize_or_crop="none"  #'scale_width_and_crop'  # "none" 'resize_and_crop' 'crop' 'scale_width' 'scale_width_and_crop' 'crop_and_scale' 'crop_and_rotate_and_scale'
preproc="None" # "suppress_half" / "None"

loadSize=400
fineSize=400

echo "start"
sleep 1
name="$loadSize"_"$fineSize"_preproc  
# NOTE: if dataroot is set to result_dir, remember to copy testA/B to the result_dir

python -u \
        test.py --dataroot $dataset_dir --result_dir $result_dir \
        --name $name \
        --resume "$model_dir" \
        --batch_size 1 \
        --serial_batches \
        --preproc $preproc\
        --loadSize $loadSize --fineSize $fineSize --resize_or_crop $resize_or_crop \

