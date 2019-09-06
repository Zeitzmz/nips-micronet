#!/bin/bash

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/scoring \
-model model/ProfitableNet.prototxt \
-sparsity model/sparsity.txt 2>&1 | tee ./log/score.log

