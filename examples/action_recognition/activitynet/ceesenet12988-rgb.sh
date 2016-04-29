#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/action_recognition/log \
    build/tools/caffe train -gpu=4,5,6,7 \
    --solver=models/action_recognition/ActivityNet/bvlc_googlenet_bottomup_12988-rgb.solver \
    --weights=data/ActivityNet/bvlc_googlenet_bottomup_12988_trainval.caffemodel
