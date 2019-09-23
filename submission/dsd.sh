#!/bin/bash

image_folder=$1
# replace imagenet train folder and validation folder
sed -i 's#root_folder: "\(.*\)train/"#root_folder: "'${image_folder}train/'"#' train/ProfitableNet.prototxt
sed -i 's#root_folder: "\(.*\)val/"#root_folder: "'${image_folder}val/'"#' train/ProfitableNet.prototxt



caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p dsd/log
mkdir -p dsd/snapshots

${caffe_tools}/caffe train \
--solver=dsd/solver.prototxt \
--weights=train/snapshots/caffe_train_iter_1500000.caffemodel \
--gpu=all 2>&1 | tee ./dsd/log/dsd_finetune.log 
