#!/bin/bash

# replace imagenet validation folder
# sed -i 's#root_folder: "\(.*\)"#root_folder: "'${image_folder}'"#' ${model}

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p dsd/log

${caffe_tools}/caffe test \
    -model dsd/ProfitableNet.prototxt \
    -weights dsd/snapshots/caffe_train_dsd_iter_400000.caffemodel \
    -iterations 2000 \
    -gpu 1 2>&1 | tee ./dsd/log/test.log 
