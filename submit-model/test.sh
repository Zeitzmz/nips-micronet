

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/caffe test \
--model=caffe_model/ProfitableNet.prototxt \
--weights=caffe_model/ProfitableNet.caffemodel \
--iterations=2000 \
--gpu=0 2>&1 | tee ./log/test.log 
