#!/bin/bash

image_folder=$1
# replace imagenet train folder and validation folder
sed -i 's#root_folder: "\(.*\)train/"#root_folder: "'${image_folder}train/'"#' sparsity/ProfitableNet.prototxt
sed -i 's#root_folder: "\(.*\)val/"#root_folder: "'${image_folder}val/'"#' sparsity/ProfitableNet.prototxt



caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p sparsity/log
mkdir -p sparsity/snapshots

${caffe_tools}/caffe train \
--solver=sparsity/solver.prototxt \
--weights=train/snapshots/caffe_train_iter_1500000.caffemodel \
--gpu=all 2>&1 | tee ./sparsity/log/sparsity_finetune.log 
