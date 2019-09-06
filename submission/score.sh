#!/bin/bash

image_folder=$1
model=model/ProfitableNet.prototxt
weights=model/ProfitableNet.caffemodel
sparsity=model/sparsity.txt

# replace imagenet validation folder
sed -i 's#root_folder: "\(.*\)"#root_folder: "'${image_folder}'"#' ${model}

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log
${caffe_tools}/scoring \
    -model ${model} \
    -sparsity ${sparsity} 2>&1 | tee ./log/score.log

