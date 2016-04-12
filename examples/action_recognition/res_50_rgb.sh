#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=4,5,6,7 \
    --solver=models/action_recognition/ResNet-50-rgb.solver \
    --weights=models/action_recognition/ResNet-50-rgb-model.caffemodel    
