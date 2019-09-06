#!/bin/bash

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/caffe test \
-model model/ProfitableNet.prototxt \
-weights model/ProfitableNet.caffemodel \
-iterations 2000 \
-gpu 0 2>&1 | tee ./log/test.log 
