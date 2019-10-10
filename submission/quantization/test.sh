#!/bin/bash

# replace imagenet validation folder
sed -i 's#root_folder: "\(.*\)"#root_folder: "'${image_folder}'"#' int8/ProfitableNet.prototxt

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/caffe test \
    -model int8/ProfitableNet.prototxt \
    -weights int8/snapshots/caffe_train_fix_point_iter_1.caffemodel \
    -iterations 2000 \
    -gpu 0 2>&1 | tee ./int8/log/test.log 
