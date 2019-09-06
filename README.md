# [MicroNet Challenge](https://micronet-challenge.github.io/)

[![Licensed under the MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Zeitzmz/nips-micronet/blob/master/LICENSE)

By Mengze Zeng, Jie Hu, Ziheng Wu from [Momenta](https://www.momenta.ai/).

All details about the challenge are documented in [submision.pdf](https://github.com/Zeitzmz/nips-micronet/blob/master/submission.pdf)


# Requirements
In this work, we take [BVLC/Caffe](https://caffe.berkeleyvision.org/) to implement our algorithms.
- install caffe environment (refer to [caffe installation](https://caffe.berkeleyvision.org/installation.html)).


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
$ cd ../submit-model
$ ./test.sh imagenet_validation_image_folder  # ./test.sh /data/imagenet1k/val/
$ ./score.sh imagenet_validation_image_folder # ./score.sh /data/imagenet1k/val/
```

# Score:
- Score: 0.25097 
- Top-1: 75.21%
- Top-5: 92.28%

