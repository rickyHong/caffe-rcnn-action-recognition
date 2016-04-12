#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=0,1,2,3 \
    --solver=models/action_recognition/ResNet-50-flow.solver \
    --weights=models/action_recognition/ResNet-50-flow-model.caffemodel    
