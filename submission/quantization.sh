                                 
caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p quantization/snapshots
mkdir -p quantization/log

${caffe_tools}/caffe train \
--solver=quantization/solver.prototxt \
--weights=sparsity/snapshots/caffe_train_sparsity_iter_400000.caffemodel \
--gpu=0 2>&1 | tee ./quantization/log/fixedpoint.log 
