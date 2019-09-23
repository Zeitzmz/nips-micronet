#!/bin/bash

image_folder=$1
# replace imagenet train folder and validation folder
sed -i 's#root_folder: "\(.*\)train/"#root_folder: "'${image_folder}train/'"#' dsd/ProfitableNet.prototxt
sed -i 's#root_folder: "\(.*\)val/"#root_folder: "'${image_folder}val/'"#' dsd/ProfitableNet.prototxt



caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools


${caffe_tools}/gen_dsd_mask \
train/snapshots/caffe_train_iter_1500000.caffemodel \
dsd/sparsity.txt \
dsd/mask.bin
