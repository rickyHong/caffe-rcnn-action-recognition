#!/usr/bin/env sh
# This script test four voc images using faster rcnn end-to-end trained model (ZF-Model)
CAFFE=build/tools/caffe

GLOG_log_dir=examples/FRCNN/log $CAFFE train   \
       --gpu 0 \
       --solver models/FRCNN/zf_rpn/solver.prototxt \
       --model models/FRCNN/ZF.v2.caffemodel 
