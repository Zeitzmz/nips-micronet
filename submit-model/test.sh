

caffe_path=../micronet-caffe
caffe_tools=${caffe_path}/build/tools

mkdir -p log

${caffe_tools}/caffe test \
--model=caffe_model/fp16.prototxt \
--weights=caffe_model/fp16.caffemodel \
--iterations 2000 \
--gpu=1 2>&1 | tee ./log/test.log 
