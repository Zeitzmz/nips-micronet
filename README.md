# [MicroNet Challenge](https://micronet-challenge.github.io/)

[![Licensed under the MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Zeitzmz/nips-micronet/blob/master/LICENSE)

By Mengze Zeng, Jie Hu, Ziheng Wu from [Momenta](https://www.momenta.ai/).

All details about the challenge are documented in [submision.pdf](https://github.com/Zeitzmz/nips-micronet/blob/master/submission.pdf)


# Requirements
In this work, we take [BVLC/Caffe](https://caffe.berkeleyvision.org/) to implement our algorithms.
- install caffe GPU environments (refer to [caffe installation](https://caffe.berkeleyvision.org/installation.html)).


# Run
1. Download project
```
$ git clone https://github.com/Zeitzmz/nips-micronet.git --recursive
```
2. Complile caffe
```
$ cd nips-micronet && git submodule foreach git pull origin master
$ cd micronet-caffe && make -j
```
3. Test submitted model & Scoring
``` 
$ cd ../submission
$ ./test.sh imagenet_validation_image_folder  # ./test.sh /data/imagenet1k/val/
# When test, we convert 'type: "DepthwiseConvolution"' to 'type: "Convolution"' in order to use fp16.
$ ./score.sh imagenet_validation_image_folder # ./score.sh /data/imagenet1k/val/
```

# Train, DSD and Int8
1. Train
```
$ ./train.sh imagenet_image_root_folder  # ./train.sh /data/imagenet1k/ (consist of dirs of train and val)
```
2. DSD
```
$ ./dsd.sh imagenet_image_root_folder  # ./dsd.sh /data/imagenet1k/ (consist of dirs of train and val)
```
3. Int8
```
$ ./int8.sh imagenet_image_root_folder  # ./int8.sh /data/imagenet1k/ (consist of dirs of train and val)
```


# Score:
- Score: 0.195493 
- Top-1: 75.0422%
- Top-5: 92.2582%

