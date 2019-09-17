#!/bin/bash

image_folder=$1
model=model/ProfitableNet.prototxt
weights=model/ProfitableNet.caffemodel
# replace imagenet validation folder
sed -i 's#root_folder: "\(.*\)"#root_folder: "'${image_folder}'"#' ${model}

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/caffe test \
    -model ${model} \
    -weights ${weights} \
    -iterations 2000 \
    -gpu 0 2>&1 | tee ./log/test.log 
