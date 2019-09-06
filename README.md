# [MicroNet Challenge](https://micronet-challenge.github.io/)

[![Licensed under the MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Zeitzmz/nips-micronet/blob/master/LICENSE)

By Mengze Zeng, Jie Hu, Ziheng Wu from [Momenta](https://www.momenta.ai/). 

In this repository, we take [BVLC/Caffe](https://caffe.berkeleyvision.org/) to implement our algorithms for ImageNet tracking.

# Requirements
- install caffe environment (please reference https://github.com/BVLC/caffe)


# Run
1.Download caffe
```
git clone https://github.com/Zeitzmz/nips-micronet.git --recursive
```
2.Complile caffe
```
cd nips-micronet && git submodule foreach git pull origin master
cd micronet-caffe && make -j
```
3.Test model
``` 
cd ../submit-model (change caffe_model/ProfitableNet.prototxt line18: root_folder: "your dir path of validation data") 
./test.sh (Testing 50000 validation images will last about 1 hour.)
```
4.Calculate model score
```
./score.sh
``` 

# Accuracy and score:
- top-1/top-5: 75.21%/92.28%
- score: 0.25097 
- More details are showed in main.pdf.
