                                 
caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p int8/snapshots
mkdir -p int8/log

${caffe_tools}/caffe train \
--solver=int8/solver.prototxt \
--weights=dsd/snapshots/caffe_train_dsd_iter_400000.caffemodel \
--gpu=0 2>&1 | tee ./int8/log/fixedpoint.log 
