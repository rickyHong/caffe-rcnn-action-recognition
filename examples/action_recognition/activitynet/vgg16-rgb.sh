#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=4,5,6,7 \
    --solver=models/action_recognition/ActivityNet/VGG16-rgb.solver \
    --weights=data/ActivityNet/VGG_ILSVRC_19_layers.caffemodel
