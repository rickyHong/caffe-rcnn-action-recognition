# Special Feature for My Caffe 

- Clone from the official caffe, will continuely be up to date by the official caffe code
- Add faster rcnn joint train and test \[DONE\]
- Add action recognition \[DONE\]
- Add spatial transform network \[Not Started\]
- Speed up\[DOING\]
- With demos including all tasks above \^_\^

## Faster RCNN End-To-End

### Disclaimer
The official Faster R-CNN code (written in MATLAB) is [available](https://github.com/ShaoqingRen/faster_rcnn) here. If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official](https://github.com/ShaoqingRen/faster_rcnn) code.

This repository contains a C++ reimplementation of the Python code([py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)). This C++ implementation is built on the offcial [caffe](https://github.com/BVLC/caffe), I will continue to update this code for improvement and up-to-date by offcial caffe.

All following steps, you should do these in the $CAFFE\_ROOT path.

### Demo
Using **sh example/FRCNN/demo.sh**, the will process five pictures in the examples/FRCNN/images , and put results into examples/FRCNN/results .

Note: You should prepare the trained caffemodel into models/FRCNN/ as ZF\_faster\_rcnn\_final.caffemodel

### Train
Using **sh example/FRCNN/train.sh**, the will start train voc2007 data using ZF model.

- VOCdevkit should be put into $CAFFE\_ROOT, ** ln -s $YOUR\_VOCdevkit\_Path $CAFFE\_ROOT/VOCdevkit
- ZF pretrain model should be put into models/FRCNN/ as ZF.v2.caffemodel

### Detail

More details in the code.

# Caffe

[![Build Status](https://travis-ci.org/BVLC/caffe.svg?branch=master)](https://travis-ci.org/BVLC/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
