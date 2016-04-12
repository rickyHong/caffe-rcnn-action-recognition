#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=4,5,6,7 \
    --solver=models/action_recognition/VGG16-rgb.solver \
    --weights=models/action_recognition/vgg_16_action_rgb_pretrain.caffemodel

