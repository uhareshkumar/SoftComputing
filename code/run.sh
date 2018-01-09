#!/bin/bash

# To run this shell script type as follows in the terminal:
#
# For training execute: ./run.sh train/skip path/to/video/file 
#       example: ./run.sh train data/sample_video.mp4 
#
# Argument: 
#         train/skip: one of the following: train, skip!
#         path/to/video/file: relative path to the video file that we want to perform lip tracking on.


if [ $# -eq 2 ]; then
    # assign the provided arguments to variables
    train_or_skip=$1
    input_filename=$2
else
    # assign the default values to variables
    train_or_skip='train'
    input_filename="/data/sample_video.mp4"
fi

if [ $train_or_skip = 'train' ]; then

    # training
    python -u ./code/training_evaluation/train.py --num_epochs=1 --batch_size=16 --train_dir=/results/TRAIN_CNN_3D/train_logs
    # testing - Automatically restore the latest checkpoint from all saved checkpoints
    python -u ./code/training_evaluation/test.py --checkpoint_dir=/results/TRAIN_CNN_3D/ --test_dir=/results/TRAIN_CNN_3D/test_logs
    
else

    echo "No training or testing will be performed!"
    
fi

# visualizing (using pretrained model)
ln -s /data dlib
python -u ./code/lip_tracking/VisualizeLip.py --input $input_filename --output ../results/output_video.mp4
mv ./activation ../results

# create gif from mouth frames
ffmpeg -i ./mouth/frame_%*.png ../results/mouth.gif