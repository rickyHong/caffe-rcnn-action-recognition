#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=0,1,2,3 \
    --solver=models/action_recognition/ActivityNet/googlenet_v1-rgb.solver \
    --weights=data/ActivityNet/bvlc_googlenet.caffemodel 
