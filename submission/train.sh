#!/bin/bash

image_folder=$1
# replace imagenet train folder and validation folder
sed -i 's#root_folder: "\(.*\)train/"#root_folder: "'${image_folder}train/'"#' train/ProfitableNet.prototxt
sed -i 's#root_folder: "\(.*\)val/"#root_folder: "'${image_folder}val/'"#' train/ProfitableNet.prototxt

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p train/log
mkdir -p train/snapshots

${caffe_tools}/caffe train \
--solver=train/solver.prototxt \
--gpu=all 2>&1 | tee ./train/log/train.log 
